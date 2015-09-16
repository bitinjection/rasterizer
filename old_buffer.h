#ifndef BUFFER
#define BUFFER

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
    : buffer(buffer),  width(width), height(height)
    {
      assert(height < std::numeric_limits<size>::max() / width);
      assert(width  < std::numeric_limits<size>::max() / height);
      //loaders::MD2Header header;

      //loaders::load_md2(header);
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

  template <template <typename> class Buffer,
           typename Pixel
           >
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
    std::vector<cml::vector3f> vertexes;
    cml::vector3f normal;
  };

  struct edge
  {
    float x_initial;
    unsigned int y_max;
    float m_inverse;
    triangle3 triangle;
  };

  struct active_edge
  {
    float x_current;
    edge e;
  };

  struct exhausted
  {
    unsigned int const row;

    explicit
    exhausted(unsigned int const row)
    : row(row) {}

    bool operator()(active_edge const ae)
    { return row == ae.e.y_max; }
  };

  bool edge_is_valid(cml::vector3f v_top, cml::vector3f v_bottom)
  {
    // TODO: review this decision and why it works
    enum {x, y};

    float const dxdy = v_bottom[y] - v_top[y];

    return !(-1.f < dxdy && dxdy < 1.f);
  }

  bool by_x(active_edge const l, active_edge const r)
  { return l.x_current < r.x_current; }

  bool by_y(cml::vector3f const l, cml::vector3f const r)
  { return l[1] < r[1]; }

  struct triangle
  {
    std::vector<cml::vector3f> vertices;
    unsigned int color;
  };

  edge create_edge(cml::vector3f const v_top, cml::vector3f const v_bottom)
  {
    enum {x, y};

    edge e;
    e.x_initial = v_top[x];
    e.y_max     = v_bottom[y];

    float const dx = v_bottom[x] - v_top[x];
    float const dy = v_bottom[y] - v_top[y];

    e.m_inverse =  dx / dy;

    return e;
  }

  void create_edges(
      triangle3 triangle,
      std::vector<std::vector<edge> >& edges,
      triangle3 original)
  {
    enum {v0, v1, v2};
    enum {x, y};

    if(edge_is_valid(triangle.vertexes[v0], triangle.vertexes[v1]))
    {
      edge e0 = create_edge(triangle.vertexes[v0], triangle.vertexes[v1]);
      e0.triangle = original;
      edges[triangle.vertexes[v0][y]].push_back(e0);
    }

    if(edge_is_valid(triangle.vertexes[v1], triangle.vertexes[v2]))
    {
      edge e1 = create_edge(triangle.vertexes[v1], triangle.vertexes[v2]);
      e1.triangle = original;
      edges[triangle.vertexes[v1][y]].push_back(e1);
    }

    if(edge_is_valid(triangle.vertexes[v0], triangle.vertexes[v2]))
    {
      edge e2 = create_edge(triangle.vertexes[v0], triangle.vertexes[v2]);
      e2.triangle = original;
      edges[triangle.vertexes[v0][y]].push_back(e2);
    }
  }

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

    cml::vector3f
    operator()(cml::vector3f const& vertex)
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

    std::vector<cml::vector3f> normals;
    for(std::vector<cml::vector3f>::iterator av = actual_vertexes.begin(); 
        av != actual_vertexes.end(); av += 3)
    {
      cml::vector3f v0 = *av;
      cml::vector3f v1 = *(av+1);
      cml::vector3f v2 = *(av+2);

      v0 = cml::transform_point(model_matrix, v0);
      v1 = cml::transform_point(model_matrix, v1);
      v2 = cml::transform_point(model_matrix, v2);

      cml::vector3f n =
      -cml::normalize(
          cml::cross(
            cml::normalize(v0 - v1),
            cml::normalize(v0 - v2)));

      normals.push_back(n);
    }
    //std::cout << normals[0] << std::endl;

    std::transform(
        actual_vertexes.begin(),
        actual_vertexes.end(),
        actual_vertexes.begin(),
        do_projection(
          model_matrix,
          view_matrix,
          projection_matrix,
          viewport_matrix));

    std::vector<triangle3> triangles;

    std::vector<cml::vector3f>::iterator n = normals.begin();
    typedef std::vector<cml::vector3f>::const_iterator verts_iter;
    for(verts_iter p = actual_vertexes.begin(); p != actual_vertexes.end(); )
    {
      triangle3 triangle;
      for(unsigned int i=0;i<3;++i)
      {
        triangle.vertexes.push_back(*p);
        ++p;
      }

      triangle.normal = *n++;

      triangles.push_back(triangle);
    }

    typedef std::vector<edge> edge_bucket;

    std::vector<triangle3> unsorted_triangles;
    unsorted_triangles.resize(triangles.size());
    std::copy(triangles.begin(), triangles.end(), unsorted_triangles.begin());

    typedef std::vector<triangle3>::iterator triangle_iter;


    enum { v0, v1, v2 };

    // edge creation
    std::vector<edge_bucket> edge_table;
    edge_table.resize(400);

    for(triangle_iter t = triangles.begin();
        t != triangles.end();
        ++t)
    {

      // cull
      cml::vector3f projected_normal = cml::cross(
          t->vertexes[1] - t->vertexes[0],
          t->vertexes[2] - t->vertexes[0]);

      if(cml::dot(projected_normal, cml::vector3f(0, 0, -1))  > 0)
      {
        std::sort(t->vertexes.begin(), t->vertexes.end(), by_y);

        assert(t->vertexes[v0][1] <= t->vertexes[v1][1]);
        assert(t->vertexes[v1][1] <= t->vertexes[v2][1]);
        assert(t->vertexes[v0][1] <= t->vertexes[v2][1]);

        //create_edges(
        //    *t,
        //    edge_table,
        //    *(unsorted_triangles.begin() + (t - triangles.begin())));
        create_edges(
            *t,
            edge_table,
            *t);
      }
    }

    typedef std::vector<edge>::iterator edge_iter; 
    typedef std::vector<active_edge>::iterator active_edge_iter; 
    typedef std::vector<edge_bucket>::iterator table_iter;

    // Ignore empty lines
    for(unsigned int i = 0u; i < edge_table.size() && edge_table[i].size() == 0; ++i);

    std::vector<active_edge> active_edges;
    for(unsigned int i = 0u; i < edge_table.size(); ++i)
    {
      // add new edges
      for(unsigned int k = 0;
          k < edge_table[i].size();
          ++k)
      {
        active_edge ae;
        ae.e = edge_table[i][k];
        ae.x_current = ae.e.x_initial;
        active_edges.push_back(ae);
      }

      std::sort(active_edges.begin(), active_edges.end(), by_x);

      bool parity = true;
      for(active_edge_iter ae = active_edges.begin();
          ae != active_edges.end();
          ++ae)
      {

        if(parity)
        {
          if((ae + 1) != active_edges.end())
          {
            segment s;
            s.x_start = ae->x_current;
            s.x_end = (ae + 1)->x_current;

            // draw segments
            cml::vector3f sp(s.x_start, i, 0.0f);
            cml::vector3f ep(s.x_end,   i, 0.0f);

            std::vector<cml::vector3f> const& verts = ae->e.triangle.vertexes;
            cml::vector3f proj_norm = cml::unit_cross(
                verts[1] - verts[0],
                verts[2] - verts[0]);
            float cull = cml::dot(proj_norm, cml::vector3f(0,0, -1));

            //if(cull > 0)
            {
              cml::vector3f light = cml::vector3f(0, 0, -1);
              //light = cml::transform_point(model_matrix, light);

              float angle = cml::dot(ae->e.triangle.normal, light);
              unsigned int adjusted = static_cast<unsigned int>(0xFF*angle);
              unsigned int color = (adjusted << 8);
              //draw_line(b, ep, sp, 0xFFFF0000);
              draw_line(b, ep, sp,color);
            }
          }
        }
        parity = !parity;
      }

      // erase exhausted edges
      active_edges.erase(
          std::remove_if(
            active_edges.begin(),
            active_edges.end(),
            exhausted(i+1)),
          active_edges.end());

      for(active_edge_iter ae = active_edges.begin();
          ae != active_edges.end();
          ++ae)
      {
        // calculate intersections
        // update current position
        ae->x_current += ae->e.m_inverse;
      }
    } 

    for(triangle_iter t = unsorted_triangles.begin();
        t != unsorted_triangles.end();
        ++t)
    {
      // backface culling
      cml::vector3f normal = cml::cross(
          t->vertexes[1] - t->vertexes[0],
          t->vertexes[2] - t->vertexes[0]);

      float cull = cml::dot(normal, cml::vector3f(0, 0, -1));
      //float cull = cml::dot(t->normal, cml::vector3f(0, 0, -1));

      //,if(cull > 0)
      {
        for(int k=0; k < actual_vertexes.size() / 3; ++k)
        {
          unsigned int offset = k*3;
          cml::vector3f tri[] = {
            actual_vertexes[indexes[offset+0]],
            actual_vertexes[indexes[offset+1]],
            actual_vertexes[indexes[offset+2]] };

          //draw_triangle(b, (t->vertexes).data(), 0xFFFFFFFF);
        }
      }
    }

  }
};

#endif
