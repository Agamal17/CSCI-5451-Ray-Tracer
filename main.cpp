//For Visual Studios
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS // For fopen and sscanf
#define _USE_MATH_DEFINES 
#endif

//Images Lib includes:
#define STB_IMAGE_IMPLEMENTATION //only place once in one .cpp file
#define STB_IMAGE_WRITE_IMPLEMENTATION //only place once in one .cpp file
#include "Include/image_lib.h" //Defines an image class and a color class


int img_width, img_height;
std::string imgName;

int main(int argc, char** argv) {
    //Read command line parameters to get scene file
    if (argc < 2) {
        std::cout << "Usage: ./a.out scenefile\n";
        return 0;
    }
    
    // TODO: Implement the parser
    // parseSceneFile(sceneFileName);

    // TODO: Define the camera parameters

    // TODO: Generate Rays and ray trace each of them: rayTrace()

    Image outputImg = Image(img_width, img_height);

    outputImg.write(imgName.c_str());
    return 0;
}