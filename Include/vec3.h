#include <cmath>

using std::sqrt;
using std::fmin;

#ifndef VEC3_H
#define VEC3_H

//Small vector library
// Represents a vector as 3 floats

struct vec3{
  float x,y,z;

  vec3(float x, float y, float z) : x(x), y(y), z(z) {}
  vec3() : x(0), y(0), z(0) {}
  vec3 operator-() const {
        return vec3(-x, -y, -z);
    }
  //Clamp each component (used to clamp pixel colors)
  vec3 clampTo1(){
    return vec3(fmin(x,1),fmin(y,1),fmin(z,1));
  }

  //Compute vector length (you may also want length squared)
  float length(){
    return sqrt(x*x+y*y+z*z);
  }

  //Create a unit-length vector
  vec3 normalized(){
    float len = sqrt(x*x+y*y+z*z);
    return vec3(x/len,y/len,z/len);
  }

};

//Multiply float and vector
//TODO - Implement: you probably also want to implement multiply vector and float ... inline vec3 operator*(float f, vec3 a)
inline vec3 operator*(float f, vec3 a){
  return vec3(a.x*f,a.y*f,a.z*f);
}

//Vector-vector dot product
inline float dot(vec3 a, vec3 b){
  return a.x*b.x + a.y*b.y + a.z*b.z;
}

//Vector-vector cross product
inline vec3 cross(vec3 a, vec3 b){
  return vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y-a.y*b.x);
}

//Vector addition
inline vec3 operator+(vec3 a, vec3 b){
  return vec3(a.x+b.x, a.y+b.y, a.z+b.z);
}

//Vector subtraction
inline vec3 operator-(vec3 a, vec3 b){
  return vec3(a.x-b.x, a.y-b.y, a.z-b.z);
}

// Useful for optimization (avoids expensive sqrt)
inline float length_squared(const vec3& v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

// Allow vec * float
inline vec3 operator*(const vec3& a, float f) {
    return vec3(a.x * f, a.y * f, a.z * f);
}

// Element-wise (Hadamard) multiplication
inline vec3 operator*(const vec3& a, const vec3& b) {
    return vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

// ----------------------------------------------------------
// Reflection
// r = v - 2*dot(v,n)*n
// ----------------------------------------------------------
inline vec3 reflect(const vec3 v, const vec3 n) {
    return v - 2.0f * dot(v, n) * n;
}

// ----------------------------------------------------------
// Refraction (Snell's Law)
// ----------------------------------------------------------
inline vec3 refract(const vec3 uv, const vec3 n, float etai_over_etat) {
    // uv: incident unit vector, n: unit normal, etai_over_etat: eta_i / eta_t
    float cos_theta = fmin(dot(-uv, n), 1.0f); 
    
    // Check for Total Internal Reflection (TIR)
    float sin_theta_squared = 1.0f - cos_theta * cos_theta;
    float discriminant = 1.0f - etai_over_etat * etai_over_etat * sin_theta_squared;
    
    if (discriminant < 0) {
        return vec3(0, 0, 0); // Signal TIR
    }

    vec3 r_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_parallel = -sqrt(discriminant) * n;
    
    return r_perp + r_parallel;
}

#endif
