import cv2 as cv
import numpy as np
import win32gui
import win32ui
import win32con
import time
import math
import keyboard
import mouse
import threading
import re
from typing import List, Tuple, Union
from itertools import repeat

"""
PyMinecraft - the tool used to control Minecraft (100% externally)
Checklist:
1. Switch to default resource pack (resource packs are theoretically acceptable, if you can find the font atlas) 
2. Toggle "Force unicode fonts: ON" in language options
3. Toggle "Raw input: OFF" in mouse options
4. Adjust GUI Scale so that the text in f3 debug screen is NOT 2x upscaled
"""

# constants
font_atlas_file = "unicode_page_00.png"
# font_atlas_file = "ascii.png"
max_line_height_deviation = 0
whitespace_width = 10
f3_color = (221, 225)
chars_per_row = 16
chars_per_col = 16

# type alias
Hwnd = int

# load font atlas
font_atlas = cv.imread(font_atlas_file, cv.IMREAD_UNCHANGED)
font_atlas = cv.threshold(font_atlas[:, :, 3], 127, 255, cv.THRESH_BINARY)[1]
char_width = font_atlas.shape[1] // chars_per_row
char_height = font_atlas.shape[0] // chars_per_col
cv.imwrite("font_atlas.bmp", font_atlas)
non_alpha = re.compile(r"[a-zA-Z]")


def crop_font(ch: str) -> np.ndarray:
    """
    Gets and crops the bitmap font of ch
    :param ch: the character
    :return: the cropped (no black border) bitmap font of ch
    """

    row = ord(ch) // chars_per_row
    col = ord(ch) % chars_per_row
    sy, sx = char_height * row, char_width * col
    ret = font_atlas[sy:sy + char_height, sx:sx + char_width]
    x1, x2 = 0, char_width
    while x1 < char_width and sum(ret[:, x1]) == 0:
        x1 += 1
    while x2 > 0 and sum(ret[:, x2 - 1]) == 0:
        x2 -= 1
    # leave 1 pixel-width black border
    return ret[:, max(x1 - 1, 0):min(x2 + 1, char_width)]


bitmap_fonts = [crop_font(chr(ch)) for ch in range(0, 128)]


def ocr_f3(cap: np.ndarray, charset: str = "") -> List[str]:
    """
    Performs simple OCR on captured f3 screen
    :param cap: the screen capture, in BGR
    :param charset: the charset, a smaller charset leaders to faster recognition
    :return: list representing rows of debug information
    """
    # constants
    match_threshold = 0.999  # "exact match"
    # convert to gray-scale image first
    img = cv.cvtColor(cap, cv.COLOR_BGR2GRAY if cap.shape[2] == 3 else cv.COLOR_BGRA2GRAY)
    # apply thresholds, so only pixels with color f3_color remains
    img = cv.threshold(img, f3_color[1], 255, cv.THRESH_TOZERO_INV)[1]
    img = cv.threshold(img, f3_color[0] - 1, 255, cv.THRESH_BINARY)[1]
    chars: List[Tuple[int, int, str]] = []  # single characters recognized
    for ch in charset:  # match characters
        font = bitmap_fonts[ord(ch)]
        x = cv.matchTemplate(img, font, cv.TM_CCORR_NORMED)
        # noinspection PyTypeChecker
        chars.extend(zip(*np.where(x > match_threshold), repeat(ch)))
    # a lot of optimizations can be done to the lines below, but they are not the bottleneck anyway
    img_width = img.shape[1]
    chars.sort(key=lambda tup: tup[0] * img_width + tup[1])
    """ for debugging
    total = np.zeros(img.shape)
    for ch in chars:
        total[ch[0], ch[1]] = 1
    cv.imshow("", total)
    cv.waitKey(0)
    """
    last_y = -1 - max_line_height_deviation
    lines: List[List[Tuple[int, int, str]]] = []
    for ch in chars:  # separate chars into multiple lines
        if ch[0] - last_y > max_line_height_deviation:
            last_y = ch[0]
            lines.append([])
        lines[-1].append(ch)
    ret: List[str] = []
    for line in lines:  # concatenate each line into string
        line_str = ""
        # there may be error in recognizing chars such as .(which may appear everywhere) or -(which _ is mistaken for)
        error = True
        for (i, ch) in enumerate(line):
            if i > 0 and ch[1] - line[i - 1][1] > whitespace_width:  # add whitespaces, if necessary
                line_str += ' '
            line_str += ch[2]
            error &= ch[2] in ['.', '-', '_']  # so if this line is full of such characters,
        if not error:  # ditch this line
            ret.append(line_str)
    return ret


def get_window_with_title(keyword: str) -> Hwnd:
    """
    Gets the first window whose title contains the keyword, if any
    !!! WIN 32 ONLY !!!
    :param keyword: the keyword
    :return: the first window whose title contains keyword, None if not found
    """

    def callback(hwnd: Hwnd, extra: List[Hwnd]):
        if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd) \
                and keyword in win32gui.GetWindowText(hwnd):
            extra.append(hwnd)

    ret: List[Hwnd] = []
    win32gui.EnumWindows(callback, ret)
    return ret[0] if ret else None


def capture_screen_of(hwnd: Hwnd, rect=None) -> np.ndarray:
    """
    Capture screen of hwnd, using win32 API
    !!! WIN 32 ONLY !!!
    :param hwnd: the window handle
    :param rect: the rectangle, default value None, which means to capture the whole window
    :return: the screen capture (RGBA) in numpy array format
    """
    window_dc = win32gui.GetWindowDC(hwnd)
    dc_object = win32ui.CreateDCFromHandle(window_dc)
    compat_dc = dc_object.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    if not rect:
        rect = win32gui.GetWindowRect(hwnd)
        rect = (0, 0, rect[2] - rect[0], rect[3] - rect[1])
    width, height = rect[2] - rect[0], rect[3] - rect[1]
    start = (rect[0], rect[1])
    bitmap.CreateCompatibleBitmap(dc_object, width, height)
    compat_dc.SelectObject(bitmap)
    compat_dc.BitBlt((0, 0), (width, height), dc_object, start, win32con.SRCCOPY)
    img = np.frombuffer(bitmap.GetBitmapBits(True), dtype='uint8')
    img.shape = (height, width, 4)
    dc_object.DeleteDC()
    compat_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, window_dc)
    win32gui.DeleteObject(bitmap.GetHandle())
    return img


class PlayerInformation(object):
    coord: np.array = None
    facing: Tuple[float, float] = None
    target_block: np.array = None


def xyz2xz(xyz: np.array) -> np.array:
    return np.array([xyz[0], xyz[2]])


def xz2xyz(xz: np.array) -> np.array:
    return np.array([xz[0], 0, xz[1]])


def get_player_info_from_f3(cap: np.ndarray) -> Union[None, PlayerInformation]:
    """
    Gets player information from f3 debug screen
    :param cap: f3 screen capture
    :return: the player information, None if the client is not in gameplay / has no f3 overlay
    """
    lines = ocr_f3(cap, "0123456789XYZFLa./-")
    if len(lines) == 0:
        return None
    ret = PlayerInformation()
    for line in lines:
        if line.startswith("XYZ"):
            xyz = non_alpha.sub('', line).split('/')
            if len(xyz) == 3:
                ret.coord = np.array([float(xyz[0]), float(xyz[1]), float(xyz[2])])
        if line.startswith("F"):
            f = non_alpha.sub('', line).split('/')
            if len(f) == 2:
                ret.facing = (float(f[0]), float(f[1]))
        if line.startswith("L a"):
            xyz = non_alpha.sub('', line).strip().split(' ')
            if len(xyz) == 3:
                ret.target_block = np.array([float(xyz[0]), float(xyz[1]), float(xyz[2])])
    return ret


player_info: Union[PlayerInformation, None] = None
last_update = time.time()
in_control = False
target_coord = np.array([0, 0])
target_yaw = 0
target_pitch = 0
reached_coord = False
reached_yaw = False
reached_pitch = False
auto_yaw = False


def daemon_thread_func(keyword, region, shrink2x=False):
    """
    The daemon thread
    :param keyword: the keyword for title of the Minecraft instance running
    :param region: the region for the thread to automatically capture
    :param shrink2x: whether to down sample captured image by 2x
    :return: this thread runs for ever
    """
    global player_info
    global last_update
    while True:
        start_time = time.time()
        window = get_window_with_title(keyword)
        f3 = capture_screen_of(window, region)
        """
        f3g = cv.cvtColor(f3, cv.COLOR_BGR2GRAY)
        f3g = cv.threshold(f3g, f3_color[1], 255, cv.THRESH_TOZERO_INV)[1]
        f3g = cv.threshold(f3g, f3_color[0] - 1, 255, cv.THRESH_BINARY)[1]
        cv.imshow("Debug", f3g)
        cv.waitKey(1)
        """
        if shrink2x:
            f3 = cv.resize(f3, (f3.shape[1] // 2, f3.shape[0] // 2), interpolation=cv.INTER_NEAREST)
        player_info = get_player_info_from_f3(f3)
        # print(1 / (time.time() - last_update))
        last_update = time.time()
        elapsed = time.time() - start_time
        time_left = max(0.0, 0.03 - elapsed)
        time.sleep(time_left)


def control_thread_func(coord_tol: float = 0.5, yaw_tol: float = 2, pitch_tol: float = 2):
    """
    The control thread. If abs(error_target) <= x_target, then the target will be considered reached
    :param coord_tol: coordinate tolerance
    :param yaw_tol: yaw tolerance
    :param pitch_tol: pitch tolerance
    :return: runs forever
    """
    kp_yaw, kp_pitch = 0.5, 0.4  # use P controller to control facing, too lazy to write PID anyway
    global target_yaw
    global target_pitch
    global reached_yaw
    global reached_pitch
    global reached_coord
    w, a, s, d = False, False, False, False  # whether the four keys are pressed
    move_threshold = 0.5  # DO NOT move in a direction if the error component in that direction is less than this
    turn_threshold = 1.5  # turn AT LEAST this much in every loop cycle, if the controller decides to move

    def key_cond(val, key, cond):
        if val and not cond:
            keyboard.release(key)
            return False
        if not val and cond:
            keyboard.press(key)
            return True
        return val

    while True:
        if not in_control or not player_info:
            time.sleep(0.05)
            continue

        yaw, pitch = player_info.facing
        coord = xyz2xz(player_info.coord)

        # coordinate control
        error_coord = math.hypot(target_coord[0] - coord[0], target_coord[1] - coord[1])
        reached_coord = error_coord < coord_tol
        theta = math.atan2(target_coord[0] - coord[0], target_coord[1] - coord[1])
        # move in ws direction
        ws = math.cos(theta + math.radians(yaw)) * error_coord
        # move in ad direction
        ad = math.sin(theta + math.radians(yaw)) * error_coord
        w = key_cond(w, "w", ws > move_threshold)
        s = key_cond(s, "s", ws < -move_threshold)
        a = key_cond(a, "a", ad > move_threshold)
        d = key_cond(d, "d", ad < -move_threshold)

        # yaw and pitch control
        target_pitch = max(-90, min(90, target_pitch))
        if auto_yaw and not reached_coord:  # automatically face the target coordinate
            target_yaw = -math.degrees(math.atan2(target_x - x, target_z - z))
        if target_yaw > 180:
            target_yaw -= 180
        if target_yaw < -180:
            target_yaw += 180
        error_yaw = target_yaw - yaw
        if abs(error_yaw - 360) < abs(error_yaw):
            error_yaw -= 360
        if abs(error_yaw + 360) < abs(error_yaw):
            error_yaw += 360
        error_pitch = target_pitch - pitch
        reached_yaw = abs(error_yaw) < yaw_tol
        reached_pitch = abs(error_pitch) < pitch_tol
        if not reached_yaw:
            d_yaw = kp_yaw * error_yaw
            if abs(d_yaw) < turn_threshold:
                d_yaw = math.copysign(turn_threshold, error_yaw)
        else:
            d_yaw = 0
        if not reached_pitch:
            d_pitch = kp_pitch * error_pitch
            if abs(d_pitch) < turn_threshold:
                d_pitch = math.copysign(turn_threshold, error_pitch)
        else:
            d_pitch = 0
        mouse.move(d_yaw, d_pitch, absolute=False)

        time.sleep(0.01)


def wait_turn():
    time.sleep(0.01)
    while player_info and (not reached_yaw or not reached_pitch):
        time.sleep(0.05)


def wait_coord():
    time.sleep(0.01)
    while player_info and not reached_coord:
        time.sleep(0.05)


daemon_thread = threading.Thread(target=daemon_thread_func, args=("MineCraft", (0, 140, 400, 350), False), daemon=True)
daemon_thread.start()
control_thread = threading.Thread(target=control_thread_func, args=(), daemon=True)
control_thread.start()


def toggle_control():
    global in_control
    in_control = not in_control


def start_mining():
    global in_control
    global target_coord
    global target_yaw
    global target_pitch
    if player_info is None:
        return
    if player_info.target_block is None:
        return
    direction = xyz2xz(player_info.target_block - np.floor(player_info.coord))
    target_coord = xyz2xz(player_info.coord)
    target_yaw = -math.degrees(math.atan2(direction[0], direction[1]))
    in_control = True
    target_pitch = 0
    moved = 0
    wait_turn()
    try:
        while True:
            moved += 1
            mouse.press()
            while np.linalg.norm(xyz2xz(player_info.target_block - np.floor(player_info.coord))) < 1.:
                time.sleep(0.01)
            mouse.release()
            if moved % 5 == 0:
                target_yaw += 30
                wait_turn()
                mouse.right_click()
                time.sleep(0.1)
                target_yaw -= 30
            target_pitch = 40
            wait_turn()
            mouse.press()
            while np.linalg.norm(xyz2xz(player_info.target_block - np.floor(player_info.coord))) < 1.:
                time.sleep(0.01)
            mouse.release()
            target_coord = xyz2xz(np.floor(player_info.coord)) + np.array([0.5, 0.5]) + 0.7 * direction
            print(target_coord)
            target_pitch = 0
            wait_turn()
            wait_coord()
    except Exception:
        in_control = False
        mouse.release()
        print("exit")
        return


# keyboard.add_hotkey("ctrl+shift+u", callback=start_mining)
keyboard.add_hotkey("ctrl+shift+u", callback=toggle_control)
while True:
    command = input("Command: ")
    if command.startswith("x"):
        target_x = float(command.replace("x", "").strip())
        print(target_x)
    if command.startswith("z"):
        target_z = float(command.replace("z", "").strip())
        print(target_z)
    if command.startswith("yaw"):
        target_yaw = float(command.replace("yaw", "").strip())
        print(target_yaw)
    if command.startswith("pitch"):
        target_pitch = float(command.replace("pitch", "").strip())
        print(target_pitch)
    if command.startswith("auto_yaw"):
        auto_yaw = not auto_yaw
        print(auto_yaw)
