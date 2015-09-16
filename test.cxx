#include <iostream>
#include <cstddef>

#include <SDL/SDL.h>

#include "buffer.h"

namespace rasterizer
{
  template <>
  struct BufferTraits<unsigned int, unsigned int**>
  {
  };
};

int
main()
{

  using namespace rasterizer;

  if(SDL_Init(SDL_INIT_VIDEO) == -1)
  { std::cout << "Could not init" << std::endl; }

  SDL_Surface* surface = SDL_SetVideoMode(550, 400, 32, SDL_SWSURFACE);
  if(surface == 0)
    std::cout << "Could not set video mode" << std::endl;

  //unsigned int* pixels = static_cast<unsigned int*>(surface->pixels);
  unsigned int* pixels = static_cast<unsigned int*>(surface->pixels);
  Buffer<unsigned int> p(pixels, 550, 400);

  cml::vector3f t[] = {
    cml::vector3f(100.0f, 100.0f, 0.f),
    cml::vector3f(50.0f, 50.0f, 0.f),
    cml::vector3f(100.0f, 50.0f, 0.f)
  };

  cml::matrix44f model_matrix = cml::identity<4>();
  matrix_set_translation(model_matrix, cml::vector3f(0, 0, -50));

  using namespace cml;

  SDL_Event event;
  bool running = true;

  while(running)
  {
    p.clear(0x00000000);
    matrix_rotate_about_local_x(model_matrix, .01f);
    project(p, model_matrix);

    SDL_Flip(surface);

    while(SDL_PollEvent(&event))
    {
      switch(event.type)
      {
        case SDL_KEYDOWN:
          running = false;
          break;
        default:
          matrix_rotate_about_local_x(model_matrix, .01f);
          break;
      }
    }

  }

}
