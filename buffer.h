#ifndef BUFFER_H
#define BUFFER_H

#include <iostream>
#include <new>

#include <numeric>
#include <vector>
#include <list>
#include <cstddef>
#include <cassert>
#include <limits>

#include <cml/cml.h>
#include <cml/mathlib/helper.h>
#include <cml/mathlib/misc.h>

//#include "md2loader.h"
#include "primitives/cube.h"


template <typename T>
struct value
{ typedef T type; } ;

template <typename T>
struct pointer
{ typedef typename value<T>::type* type; };

template <typename T>
struct reference
{ typedef typename value<T>::type& type; };

namespace rasterizer
{
  typedef std::size_t position;
  typedef std::size_t size;

  template <typename Pixel, typename DeviceBuffer>
  struct BufferTraits
  {
    static void render(
        Pixel* in_buf,
        size s,
        DeviceBuffer** out_buffer);
  };

  template <typename Pixel>
  struct Buffer
  {

    struct pixel_value
    { typedef typename value<Pixel>::type   type; };

    struct pixel_pointer
    { typedef typename pointer<Pixel>::type type; };

  public:

    /**
     * @param buffer  must exist for the entire lifetime of this object.
     * @pre \f$ width*height < numeric\_limits<size>::max() \f$.
     * @pre buffer must point to an allocated block of memory which can hold
     *      \f$ width*height \f$ Pixels.
     */
    Buffer(typename pixel_pointer::type buffer, size width, size height)
    : buffer(buffer), width(width), height(height)
    {

      assert(height < std::numeric_limits<size>::max() / width);
      assert(width  < std::numeric_limits<size>::max() / height);

    }


    /**
     * @pre \f$ x + y * width < width * height \f$
     */
    void operator()(position x, position y, Pixel p)
    {

      assert(x + y * this->width < this->width * this->height);

      *(this->buffer + x + y * this->width) = p;

    }

    void clear(Pixel c)
    {

      for(size i = 0; i < width; ++i)
        for(size j = 0; j < height; ++j)
          *(this->buffer + i + j * this->width) = c;

    }

  private:
    Buffer(Buffer<Pixel> const& other) { }
    Buffer& operator=(Buffer const& other) { }


  private:
    Pixel* buffer;
    size   width;
    size   height;
  };

  template <template <typename> class Buffer, typename Pixel>
  void draw_triangle(Buffer<Pixel>& b, cml::vector3f* t, Pixel c)
  {

    enum { p0, p1, p2 };

    draw_line(
        b,
        *(t+p0),
        *(t+p1),
        c);

    draw_line(
        b,
        *(t+p1),
        *(t+p2),
        c);

    draw_line(
        b,
        *(t+p2),
        *(t+p0),
        c);

  }


  template <template <typename> class Buffer, typename Pixel>
  void draw_line(
      Buffer<Pixel>& buffer,
      cml::vector3f p0,
      cml::vector3f p1,
      Pixel c)
  {

    enum { x, y };

    bool const steep = abs(p1[y] - p0[y]) > abs(p1[x] - p0[x]);

    if (steep)
    {
      using std::swap;

      swap(p0[x], p0[y]);
      swap(p1[x], p1[y]);
    }

    if (p0[x] > p1[x])
    {
      using std::swap;

      swap(p0[x], p1[x]);
      swap(p0[y], p1[y]);
    }

    int const dx    = p1[x] - p0[x];
    int const dy    = abs(p1[y] - p0[y]);
    int const ystep = (p0[y] < p1[y]) ? 1 : -1;

    int error = dx / 2;
    int yc    = p0[y];

    for (int xc = p0[x]; xc <= p1[x]; ++xc)
    {
      if(steep) buffer(yc, xc, c);
      else      buffer(xc, yc, c);

      error -= dy;

      if (error < 0)
      {
        yc    += ystep;
        error += dx;
      }
    }

  }

  struct segment
  {

    unsigned int x_start;
    unsigned int x_end;
    unsigned int z_start;
    unsigned int z_end;

  };

  struct triangle3
  {

    std::vector<cml::vector3f> v;
    cml::vector3f n;

  };

  struct do_indexing
  {
    std::vector<cml::vector3f> const& vertexes;

    do_indexing(std::vector<cml::vector3f> const& vertexes)
    : vertexes(vertexes) {}

    cml::vector3f
    operator()(unsigned int index)
    { return vertexes[index]; }
  };

  struct do_projection
  {

    cml::matrix44f const& model_matrix;
    cml::matrix44f const& view_matrix;
    cml::matrix44f const& projection_matrix;
    cml::matrix44f const& viewport_matrix;

    do_projection(
        cml::matrix44f const& model_matrix,
        cml::matrix44f const& view_matrix,
        cml::matrix44f const& projection_matrix,
        cml::matrix44f const& viewport_matrix)
    : model_matrix(model_matrix),
    view_matrix(view_matrix),
    projection_matrix(projection_matrix),
    viewport_matrix(viewport_matrix) {}

    cml::vector3f operator()(cml::vector3f const& vertex)
    { 

      return project_point(
          model_matrix,
          view_matrix,
          projection_matrix,
          viewport_matrix,
          vertex);

    }

  };

  inline
  cml::vector3f calculate_normal(triangle3 const& t)
  {

    cml::vector3f n =
    -cml::normalize(
        cml::cross(
          cml::normalize(t.v[0] - t.v[1]),
          cml::normalize(t.v[0] - t.v[2])));

    return n;

  }

  inline
  triangle3 create_triangle(cml::vector3f const* vertexes)
  {

    triangle3 t;

    for(unsigned int i=0; i < 3; ++i)
    {
      t.v.push_back(*vertexes++);
    }

    t.n = calculate_normal(t);

    return t;

  }

  std::vector<triangle3> create_triangles(std::vector<cml::vector3f> vertexes)
  {

    assert(vertexes.size() % 3 == 0);

    std::vector<triangle3> triangles;

    for(int k = 0; k < vertexes.size(); k+=3)
    {
      triangle3 t = create_triangle(&(vertexes[k]));
      triangles.push_back(t);
    }

    return triangles;

  }

  struct edge
  {

    float x_initial;
    float y_initial;
    float y_maximum;
    float dxdy;

  };

  template <typename Vertex> inline
  edge create_edge(Vertex const& top, Vertex const& bottom)
  {

    enum {x, y, z};

    float dy = (bottom[y] - top[y]);

    edge e;
    e.x_initial = top[x];
    e.y_initial = top[y];
    e.y_maximum = bottom[y];
    if(dy > 0)
      e.dxdy = (bottom[x] - top[x]) / dy;
    else
      e.dxdy = 0.f;

    return e;

  }

  //template <typename Vertex> inline
  bool by_y(cml::vector3f const& l, cml::vector3f const& r)
  { return l[1] < r[1]; }

  void fill_spans(Buffer<unsigned int>& b, edge e0, edge& e1, unsigned int color)
  {

    unsigned int scanline = e0.y_initial;
    float x_start = e0.x_initial;
    float x_end = e1.x_initial;

    while(scanline < e0.y_maximum-1)
    {
      draw_line(
          b,
          cml::vector3f(x_start, scanline, 0),
          cml::vector3f(x_end, scanline, 0),
          color);
      x_start += e0.dxdy;
      x_end   += e1.dxdy;
      ++scanline;
    }

    e1.x_initial = x_end;

  }

  void project(Buffer<unsigned int>& b, cml::matrix44f& model_matrix)
  {

    cml::matrix44f projection_matrix;
    cml::matrix44f view_matrix = cml::identity<4>();
    cml::matrix44f viewport_matrix;

    cml::matrix_viewport(
        viewport_matrix,
        0.f,
        550.f,
        400.f,
        0.f,
        cml::z_clip_neg_one,
        -1.f,
        1.f);

    cml::matrix_perspective(
        projection_matrix,
        1.0f,
        1.0f,
        1.0f, 
        100.0f,
        cml::right_handed,
        cml::z_clip_neg_one);

    enum { x, y, z };

    std::vector<cml::vector3f> vertexes;
    std::vector<unsigned int> indexes;

    primitives::generate_vertexes(vertexes);
    primitives::generate_indexes(indexes);

    std::vector<cml::vector3f> actual_vertexes;
    actual_vertexes.resize(indexes.size());
    std::transform(
        indexes.begin(),
        indexes.end(),
        actual_vertexes.begin(),
        do_indexing(vertexes));

    std::vector<triangle3> unprojected_triangles = create_triangles(actual_vertexes);

    for(int i = 0; i < unprojected_triangles.size(); ++i)
    {

      unprojected_triangles[i].v[0] = transform_vector(model_matrix, unprojected_triangles[i].v[0]);
      unprojected_triangles[i].v[1] = transform_vector(model_matrix, unprojected_triangles[i].v[1]);
      unprojected_triangles[i].v[2] = transform_vector(model_matrix, unprojected_triangles[i].v[2]);

      unprojected_triangles[i].n = calculate_normal(unprojected_triangles[i]);
    }

    std::transform(
        actual_vertexes.begin(),
        actual_vertexes.end(),
        actual_vertexes.begin(),
        do_projection(
          model_matrix,
          view_matrix,
          projection_matrix,
          viewport_matrix));

    std::vector<triangle3> projected_triangles = create_triangles(actual_vertexes);

    typedef std::vector<triangle3>::iterator triangle_iter;

    for(triangle_iter t = projected_triangles.begin();
        t != projected_triangles.end();
        ++t)
    {

      cml::vector3f light = cml::vector3f(0, 0, -1);
      float angle = cml::dot(cml::normalize(unprojected_triangles[t - projected_triangles.begin()].n), light);
      unsigned int adjusted = static_cast<unsigned int>(0xFF*angle);
      unsigned int color = (adjusted << 8);
      //color = 0xFFFF0000;


      if(0.f > cml::dot(cml::vector3f(0, 0, -1),t->n))
      {
        std::sort(t->v.begin(), t->v.end(), by_y);

        std::vector<edge> edges;
        edges.push_back(create_edge(t->v[0], t->v[1]));
        edges.push_back(create_edge(t->v[1], t->v[2]));
        edges.push_back(create_edge(t->v[0], t->v[2]));

        float y_min = edges[0].y_initial;
        float y_max = edges[2].y_maximum;
        float x_start = edges[0].x_initial;
        float x_end           = edges[2].x_initial;
        unsigned int scanline = edges[0].y_initial;

        fill_spans(b, edges[0], edges[2], 0xFFFF0000);
        fill_spans(b, edges[1], edges[2], 0x000000FF);

    }

    for(triangle_iter t = projected_triangles.begin();
        t != projected_triangles.end();
        ++t)
    {
      float cull = cml::dot(t->n, cml::vector3f(0, 0, -1));
      //float cull = cml::dot(t->normal, cml::vector3f(0, 0, -1));

      if(cull < 0)
      {
        cml::vector3f tri[] = {
          t->v[0],
          t->v[1],
          t->v[2]
        };

        //draw_triangle(b, (t->v).data(), 0xFFFFFFFF);
      }
    }

  }

};

#endif
