# Augmented Reality Paint

Augmented reality paint consists of a couple of scripts coded to enable the user to paint on a screen or image, using a detected object.

First, the user must create a JSON file with color limits, using the script color_segmenter.py. 
After running the script, change the trackbars until you isolate only the object that you want to use as a pen. It is highly recommended that your object has just one color to better detachment.
When you achieve the desired image, just press 'w' to create the JSON file. The file will be saved in the script folder by the name limits.json. 
Press 'q' at any time to close without saving. 

With the JSON file created, run the script ar_paint.py and provide the JSON file with the command:

    ./ar_paint.py -j limits.json (or full path to JSON file)

A window will pop up on the screen and, if your camera is working properly, you will be able to draw on the screen using the given object.
To improve accuracy on the movements of the pen, use the command '-usp' on the terminal. This will prevent long-distance moves from the pen.

To load an image to paint on, run the command:

    ./ar_paint.py -j limits.json -usp -im BLOB3_0.png

After finishing the painting just press 'w' to save and evaluate the accuracy of the painting. Note that you can choose between 4 different predefined images, for that, just replace BLOOB3_0 for BLOOB4_0, BLOOB5_0, or BLOOB6_0. In this mode, you can instead use the mouse to paint if you prefer.
To load a different image to paint on, just put the full path to the image after '-im' although, the painting will not be evaluated. 
For this mode, always use the command '-usp' or you will get an error.

Finally, the script allows the user to unpaint a given image by running on the terminal:

    ./ar_paint.py -j limits.json -usp -ip (full path to image)

The colors on the image will be removed and you will be able to paint it back, using the object or the mouse.

Here are the interactive keys:
-
Basic Letters:
-
'w' - save image;

'q' - Closes the program;

'r' - Sets the drawing pencil color to RED;

'g' - Sets the drawing pencil color to GREEN;

'b' - Sets the drawing pencil color to BLUE;

'+' - Increases the drawing pencil thickness;

'-' - Decreases the drawing pencil thickness;

'c' - Clears the drawing screen;

Extra Letters:
-

'm' - Sets the drawing pencil default color;

'a' - Allows the user to clean the screen with the pointer, like an eraser;

'f' - Allows the user to switch between a black and white backgrounds;

'p' - Enables the pointer mode i.e allows the user to move a pointer on the screen;

'j' - Draws a rectangle using mouse events;

'o' - Draws a circle using mouse events;

's' - Draws a rectangle;

'e' - Draws a circle;

'l' - The use of the following instructions ('s', 'e', 'j' and 'o') requires the pressing of the letter 'l' to draw the object with the desired dimentions
