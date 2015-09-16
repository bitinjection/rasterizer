#ifndef CUBE_H
#define CUBE_H

#include <vector>

#include <cml/cml.h>

namespace primitives
{
  void
  generate_indexes(std::vector<unsigned int>& indexes)
  {
    indexes.push_back(0);indexes.push_back(1);indexes.push_back(3);
    indexes.push_back(1);indexes.push_back(2);indexes.push_back(3);

    indexes.push_back(1);indexes.push_back(5);indexes.push_back(2); 
    indexes.push_back(5);indexes.push_back(6);indexes.push_back(2); 

    indexes.push_back(5);indexes.push_back(4);indexes.push_back(7);
    indexes.push_back(7);indexes.push_back(6);indexes.push_back(5);

    indexes.push_back(7);indexes.push_back(0);indexes.push_back(3);
    indexes.push_back(7);indexes.push_back(4);indexes.push_back(0);

    indexes.push_back(0);indexes.push_back(4);indexes.push_back(1); 
    indexes.push_back(1);indexes.push_back(4);indexes.push_back(5);

    indexes.push_back(2);indexes.push_back(7);indexes.push_back(3);
    indexes.push_back(2);indexes.push_back(6);indexes.push_back(7);
  }

  void
  generate_vertexes(std::vector<cml::vector3f>& vertexes)
  {
    vertexes.push_back(cml::vector3f(5.f  , 5.f  ,5.f));
    vertexes.push_back(cml::vector3f(-5.f , 5.f  ,5.f));
    vertexes.push_back(cml::vector3f(-5.f , -5.f ,5.f));
    vertexes.push_back(cml::vector3f(5.f  , -5.f ,5.f));

    vertexes.push_back(cml::vector3f(5.f  , 5.f  ,-5.f));
    vertexes.push_back(cml::vector3f(-5.f , 5.f  ,-5.f));
    vertexes.push_back(cml::vector3f(-5.f , -5.f ,-5.f));
    vertexes.push_back(cml::vector3f(5.f  , -5.f ,-5.f));
  };
};

#endif
