# PyMinecraft

An attempt to automate Minecraft gameplay non-intrusively.

## How does it work?

Basically the program continuously 

* Captures screenshot of the Minecraft window
* Expects the user to have enabled debug overlay (F3)
* Separates debug overlay text
* Performs OCR and reads debug information
* Decides what to do (this can be customized)
* Simulates mouse and keyboard events

## Requirements

To use this program, you'll need to 
* Install Python 3.7 and relevant packages using `pip`.
* Have a Minecraft version that supports debug overlay.
* Better switch to the default resource pack, or a resource pack that does not alter
GUI font, or whatever resourcepack you like as long as you can find the font atlas image.
* Toggle "Force Unicode Fonts: ON" in language options.
* Toggle "Raw Input: OFF" in mouse options.
* Adjust GUI scale so that the debug overlay text isn't 2x upscaled (Use the smallest GUI scale possible).

## What it currently does

The program currently reads only XYZ coordinates and camera yaw / pitch.
A closed loop control routine is implemented. Coordinate movement is controlled using a simple deadband controller and 
facing is controlled using a simple P controller with deadband.

The program also now offers a simple REPL-like interface.
* Input `x/z VALUE` sets the target coordinate component to the specified value.
* Input `yaw/pitch VALUE` sets the target camera orientation.
* Input `auto_yaw` toggles the auto-yaw function, which automatically turns the player to the target XZ coordinate while moving 
(so the player doesn't move like a crab and it's a lot faster). 

Finally, these close-loop control can be toggled in game using key `Ctrl+Shift+U`.

In addition, there is a function `start_mining` that controls the player to 
dig a 1x2 tunnel and light it up with torches (given they're in offhand).

## Strength

This program runs smoothly with vanilla Minecraft. No patches, no forge, no modification to jar files, no internal hooks. 
So in theory it has the best cross-version compatibility.

## Limitations

* This program has no intention to disguise itself. It is intended to be a proof-of-concept instead of a cheat program. 
One can recognize its operation simply from analyzing mouse movement patterns.
* The program currently gets all information from the debug overlay. It cannot get what is not 
on the F3 screen.
* The program is a bit CPU demanding, since it tries to do OCR real time.

## About OCR

I do not use Tesseract for OCR in this program because it is too slow and it does not 
perform well with pixel fonts by default without extra training (which I am too lazy to do). The OCR
algorithm I use here simply extracts font pixels from the atlas image on initialization 
and perform template matching. This works with forced unicode fonts but seems
to break with the default ASCII font.
