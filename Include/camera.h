#pragma once

#include "scene.h"
#include "ray.h"

Ray generateCameraRay(const Scene &scene,
                      int img_width,
                      int img_height,
                      int i,
                      int j);
