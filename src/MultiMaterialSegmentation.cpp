// Force cmath to include M_PI
#define _USE_MATH_DEFINES
#include "MultiMaterialSegmentation.h"
#include "Application.h"
#include "BoostInterface.hpp"
#include "Scene.h"
#include "Slice.h"
#include "boost/polygon/voronoi.hpp"
#include "utils/Constant.h"
#include "utils/Simplify.h"
#include "utils/ThreadPool.h"
#include "utils/linearAlg2D.h"
#include "utils/polygonUtils.h"

namespace cura
{

class Slice;

void MultiMaterialSegmentation::paintingSlicer(Slicer* slicer, Slicer* color_slicer)
{
    MultiMaterialSegmentation mms(slicer->layers.size());
    mms.paintingSlicerLayers(slicer, color_slicer);
}

void MultiMaterialSegmentation::paintingSlicerLayers(Slicer* slicer, Slicer* color_slicer)
{
    auto& layers = slicer->layers;
    auto* p_mesh = slicer->mesh;

    for (int i = 0; i < layers.size(); ++i)
    {
        layers[i].polygons_ = layers[i].polygons_.unionPolygons();
        layers[i].polygons_ = Simplify(100, 10, 10000).polygon(layers[i].polygons_);
        color_slicer->layers.emplace_back();
        color_slicer->layers[i].z_ = layers[i].z_;
    }

    int top_layers = std::max(slicer->mesh->settings_.get<int>("top_layers"), 1);
    int bottom_layers = std::max(slicer->mesh->settings_.get<int>("bottom_layers"), 1);
    coord_t wall_outer_line_width = slicer->mesh->settings_.get<coord_t>("wall_line_width_0");
    coord_t wall_inner_line_width = slicer->mesh->settings_.get<coord_t>("wall_line_width_x");

    coord_t z_h = MM2INT(1);
    coord_t min_offset_len = MM2INT(0.04);

    std::vector<Shape> no_colored_top_faces_polys_list(layers.size());
    std::vector<Shape> no_colored_bottom_faces_polys_list(layers.size());

    cura::parallel_for<int>(0,
                            (int)layers.size(),
                            [&](int layer_nr)
                            {
                                m_colored_lines_polys_list[layer_nr] = paintingSlicerLayerColoredLines(slicer->layers[layer_nr]);

                                Shape top_diff_polys = layer_nr < (layers.size() - 1) ? layers[layer_nr].polygons_.difference(layers[layer_nr + 1].polygons_) : layers[layer_nr].polygons_;
                                Shape bottom_diff_polys = layer_nr > 0 ? layers[layer_nr].polygons_.difference(layers[layer_nr - 1].polygons_) : layers[layer_nr].polygons_;

                                coord_t top_z = layer_nr < (layers.size() - 1) ? layers[layer_nr + 1].z_ : layers[layer_nr].z_ + z_h;
                                coord_t bottom_z = layer_nr > 0 ? layers[layer_nr - 1].z_ : layers[layer_nr].z_ - z_h;

                                // Top
                                Shape colored_top_faces_polys = paintingSlicerLayerColoredFaces(layers[layer_nr], p_mesh, layers[layer_nr].z_, top_z);
                                m_colored_top_faces_polys_list[layer_nr] = colored_top_faces_polys.intersection(top_diff_polys);
                                no_colored_top_faces_polys_list[layer_nr] = top_diff_polys.difference(colored_top_faces_polys);

                                // Bottom
                                Shape colored_bottom_faces_polys = paintingSlicerLayerColoredFaces(layers[layer_nr], p_mesh, bottom_z, layers[layer_nr].z_);
                                m_colored_bottom_faces_polys_list[layer_nr] = colored_bottom_faces_polys.intersection(bottom_diff_polys);
                                no_colored_bottom_faces_polys_list[layer_nr] = bottom_diff_polys.difference(colored_bottom_faces_polys);
                            });

    int top_parallel_size = std::ceil(static_cast<double>(layers.size()) / top_layers);
    int bottom_parallel_size = std::ceil(static_cast<double>(layers.size()) / bottom_layers);

    std::vector<Shape> colored_skin_faces_polys_list(layers.size());
    std::vector<Shape> no_colored_skin_faces_polys_list(layers.size());

    for (int i = 0; i < top_layers; ++i)
    {
        cura::parallel_for<int>(0,
                                top_parallel_size,
                                [&](int j)
                                {
                                    int layer_nr = j * top_layers + i;
                                    if (layer_nr >= layers.size())
                                    {
                                        return;
                                    }

                                    m_colored_faces_polys_list[layer_nr].push_back(m_colored_top_faces_polys_list[layer_nr]);

                                    Shape intersection_outline_polys = layers[layer_nr].polygons_;
                                    coord_t width = -wall_outer_line_width;
                                    for (int k = layer_nr - 1; k >= std::max(0, layer_nr - top_layers + 1); --k)
                                    {
                                        intersection_outline_polys = intersection_outline_polys.intersection(layers[k].polygons_);

                                        Shape offset_polys = intersection_outline_polys.offset(width);

                                        Shape skin_polys = m_colored_top_faces_polys_list[layer_nr].intersection(offset_polys);
                                        skin_polys = PolygonUtils::simplifyByScale(skin_polys, wall_outer_line_width);
                                        colored_skin_faces_polys_list[k].push_back(skin_polys);

                                        Shape no_skin_polys = no_colored_top_faces_polys_list[layer_nr].intersection(offset_polys);
                                        no_skin_polys = PolygonUtils::simplifyByScale(no_skin_polys, wall_outer_line_width);
                                        no_colored_skin_faces_polys_list[k].push_back(no_skin_polys);

                                        width -= wall_inner_line_width;
                                    }
                                });
    }

    for (int i = 0; i < top_layers; ++i)
    {
        cura::parallel_for<int>(0,
                                bottom_parallel_size,
                                [&](int j)
                                {
                                    int layer_nr = j * top_layers + i;
                                    if (layer_nr >= layers.size())
                                    {
                                        return;
                                    }
                                    m_colored_faces_polys_list[layer_nr].push_back(m_colored_bottom_faces_polys_list[layer_nr]);

                                    Shape intersection_outline_polys = layers[layer_nr].polygons_;
                                    coord_t width = -wall_outer_line_width;
                                    for (int k = layer_nr + 1; k <= std::min((int)layers.size() - 1, layer_nr + top_layers - 1); ++k)
                                    {
                                        intersection_outline_polys = intersection_outline_polys.intersection(layers[k].polygons_);

                                        Shape offset_polys = intersection_outline_polys.offset(width);

                                        Shape skin_polys = m_colored_bottom_faces_polys_list[layer_nr].intersection(offset_polys);
                                        skin_polys = PolygonUtils::simplifyByScale(skin_polys, wall_outer_line_width);
                                        colored_skin_faces_polys_list[k].push_back(skin_polys);

                                        Shape no_skin_polys = no_colored_bottom_faces_polys_list[layer_nr].intersection(offset_polys);
                                        no_skin_polys = PolygonUtils::simplifyByScale(no_skin_polys, wall_outer_line_width);
                                        no_colored_skin_faces_polys_list[k].push_back(no_skin_polys);

                                        width -= wall_inner_line_width;
                                    }
                                });
    }

    cura::parallel_for<int>(0,
                            (int)layers.size(),
                            [&](int layer_nr) {
                                Shape offset_polys = layers[layer_nr].polygons_.offset(-wall_outer_line_width);

                                colored_skin_faces_polys_list[layer_nr] = colored_skin_faces_polys_list[layer_nr].unionPolygons();
                                no_colored_skin_faces_polys_list[layer_nr] = no_colored_skin_faces_polys_list[layer_nr].unionPolygons();

                                colored_skin_faces_polys_list[layer_nr] = colored_skin_faces_polys_list[layer_nr].intersection(offset_polys);
                                no_colored_skin_faces_polys_list[layer_nr] = no_colored_skin_faces_polys_list[layer_nr].intersection(offset_polys);

                                m_colored_faces_polys_list[layer_nr] = m_colored_faces_polys_list[layer_nr].unionPolygons(colored_skin_faces_polys_list[layer_nr]);
                            });

    cura::parallel_for<int>(0,
                            (int)layers.size(),
                            [&](int layer_nr)
                            {
                                Shape offset_polys = layers[layer_nr].polygons_.offset(-wall_outer_line_width);

                                m_colored_lines_polys_list[layer_nr] = m_colored_lines_polys_list[layer_nr].difference(no_colored_skin_faces_polys_list[layer_nr]);

                                Shape top_diff_polys = layer_nr < (layers.size() - 1) ? layers[layer_nr].polygons_.difference(layers[layer_nr + 1].polygons_) : layers[layer_nr].polygons_;
                                Shape no_colored_top_skin_polys = top_diff_polys.difference(m_colored_top_faces_polys_list[layer_nr].offset(min_offset_len));
                                no_colored_top_skin_polys = no_colored_top_skin_polys.difference(m_colored_faces_polys_list[layer_nr]);

                                m_colored_lines_polys_list[layer_nr] = m_colored_lines_polys_list[layer_nr].difference(no_colored_top_skin_polys.intersection(offset_polys));
                                m_colored_lines_polys_list[layer_nr] = m_colored_lines_polys_list[layer_nr].difference(PolygonUtils::simplifyByScale(no_colored_top_skin_polys, wall_outer_line_width));


                                Shape bottom_diff_polys = layer_nr > 0 ? layers[layer_nr].polygons_.difference(layers[layer_nr - 1].polygons_) : layers[layer_nr].polygons_;
                                Shape no_colored_bottom_skin_polys = bottom_diff_polys.difference(m_colored_bottom_faces_polys_list[layer_nr].offset(min_offset_len));
                                no_colored_bottom_skin_polys = no_colored_bottom_skin_polys.difference(m_colored_faces_polys_list[layer_nr]);

                                m_colored_lines_polys_list[layer_nr] = m_colored_lines_polys_list[layer_nr].difference(no_colored_bottom_skin_polys.intersection(offset_polys));
                                m_colored_lines_polys_list[layer_nr] = m_colored_lines_polys_list[layer_nr].difference(PolygonUtils::simplifyByScale(no_colored_bottom_skin_polys, wall_outer_line_width));


                                if (m_colored_lines_polys_list[layer_nr].size() > 1)
                                {
                                    std::vector<Shape> split_polys;
                                    Shape res;
                                    PolygonUtils::splitToSimplePolygons(m_colored_lines_polys_list[layer_nr], split_polys);

                                    for (int j = 0; j < split_polys.size(); ++j)
                                    {
                                        if (split_polys[j].difference(offset_polys).empty())
                                        {
                                            continue;
                                        }
                                        res.push_back(split_polys[j]);
                                    }
                                    m_colored_lines_polys_list[layer_nr] = res;
                                }
                            });

    cura::parallel_for<int>(0,
                            (int)layers.size(),
                            [&](int layer_nr)
                            {
                                Shape colored_polys = m_colored_faces_polys_list[layer_nr].unionPolygons(m_colored_lines_polys_list[layer_nr]);
                                Shape no_colored_polys = layers[layer_nr].polygons_.difference(colored_polys.offset(20));
                                color_slicer->layers[layer_nr].polygons_ = colored_polys;
                                layers[layer_nr].polygons_ = no_colored_polys;
                            });
}

Shape MultiMaterialSegmentation::paintingSlicerLayerColoredLines(SlicerLayer& slicer_layer)
{
    if (! slicer_layer.hasColoredSegments())
    {
        return Shape();
    }
    Shape old_polygons = slicer_layer.polygons_;
    if (slicer_layer.color_segments_set_.size() == 1)
    {
        return old_polygons;
    }

    std::vector<SlicerSegment> slicer_segments;
    std::copy_if(slicer_layer.segments_.begin(), slicer_layer.segments_.end(), std::back_inserter(slicer_segments), [](SlicerSegment& slicer_segment) { return slicer_segment.color == MESH_PAINTING_COLOR; });

    if (slicer_segments.size() == 0)
    {
        return Shape();
    }

    OpenLinesSet color_line_polys_raw;
    for (int i = 0; i < slicer_segments.size(); ++i)
    {
        Polygon poly;
        auto slicer_segment = slicer_segments[i];
        poly.push_back(slicer_segment.start);
        poly.push_back(slicer_segment.end);
        color_line_polys_raw.push_back(poly.toPseudoOpenPolyline());
    }

    coord_t min_offset_len = MM2INT(0.04);
    Shape color_line_polys = color_line_polys_raw.offset(min_offset_len);

    Shape all_voronoi_color_polys;
    std::vector<Shape> simple_polygons;

    PolygonUtils::splitToSimplePolygons(old_polygons, simple_polygons);
    for (int i = 0; i < simple_polygons.size(); ++i)
    {
        Shape polys = simple_polygons[i];

        Shape color_polys;
        std::vector<Segment> color_segments;
        coloredLineSegmentMatching2(polys, color_line_polys, color_polys, color_segments);
//        coloredLineSegmentMatching(polys, color_line_polys, color_polys, color_segments);

        std::set<int> colors;
        for (int j = 0; j < color_segments.size(); ++j)
        {
            colors.insert(color_segments[j].color);
        }

        if (colors.size() == 1)
        {
            if (colors.find(MESH_PAINTING_COLOR) != colors.end())
            {
                all_voronoi_color_polys.push_back(polys);
            }
            continue;
        }

        Shape voronoi_color_polys = toVoronoiColorPolygons(color_segments);
        voronoi_color_polys = voronoi_color_polys.intersection(polys);
        all_voronoi_color_polys.push_back(voronoi_color_polys);
    }

    all_voronoi_color_polys = all_voronoi_color_polys.unionPolygons();
    all_voronoi_color_polys = all_voronoi_color_polys.intersection(old_polygons);

    return all_voronoi_color_polys;
}

Shape MultiMaterialSegmentation::toVoronoiColorPolygons(std::vector<Segment>& colored_segments)
{
    vd_t vonoroi_diagram;
    construct_voronoi(colored_segments.begin(), colored_segments.end(), &vonoroi_diagram);

    Shape voronoi_color_polys;

    for (int i = 0; i < vonoroi_diagram.cells().size(); ++i)
    {
        auto& cell = vonoroi_diagram.cells()[i];
        if (! cell.incident_edge() && ! cell.contains_segment())
        {
            continue;
        }
        auto& segment = colored_segments[cell.source_index()];
        if (segment.color <= MESH_NO_PAINTING_COLOR)
        {
            continue;
        }

        Polygon poly;
        auto from = segment.from();
        auto to = segment.to();

        auto* start = cell.incident_edge();
        bool is_circle = false;
        do
        {
            if (start->vertex0() == nullptr && start->vertex1())
            {
                break;
            }
            start = start->prev();

            if (start == cell.incident_edge())
            {
                is_circle = true;
            }
        } while (start != cell.incident_edge());

        if (! is_circle)
        {
            poly.push_back(to);
        }
        auto* next = start;
        do
        {
            if (next->vertex0() && next->vertex1() == nullptr)
            {
                break;
            }
            poly.push_back(Point2LL(next->vertex1()->x(), next->vertex1()->y()));
            next = next->next();
        } while (next != start);
        if (! is_circle)
        {
            poly.push_back(from);
        }
        poly.sortArea();
        voronoi_color_polys.push_back(poly);
    }

    voronoi_color_polys = voronoi_color_polys.unionPolygons();

    return voronoi_color_polys;
}


void MultiMaterialSegmentation::coloredLineSegmentMatching(Shape& polys, Shape& color_line_polys, Shape& out_color_polys, std::vector<Segment>& out_color_segments)
{
    AABB poly_lines_aabb;
    for (int i = 0; i < color_line_polys.size(); ++i)
    {
        for (int j = 0; j < color_line_polys[i].size(); ++j)
        {
            poly_lines_aabb.include(color_line_polys[i][j]);
        }
    }

    for (int i = 0; i < polys.size(); ++i)
    {
        Polygon color_poly;
        for (int j = 0; j < polys[i].size(); ++j)
        {
            Point2LL& p1 = polys[i][j];
            Point2LL& p2 = polys[i][(j + 1) % polys[i].size()];

            AABB aabb;
            aabb.include(p1);
            aabb.include(p2);

            if (! poly_lines_aabb.hit(aabb))
            {
                out_color_segments.emplace_back(&out_color_polys, out_color_polys.size(), color_poly.size(), MESH_NO_PAINTING_COLOR);
                color_poly.emplace_back(p1);
                continue;
            }

            Line res = linePolygonsIntersection(p1, p2, color_line_polys);

            for (int k = 0; k < res.colors.size(); ++k)
            {
                out_color_segments.emplace_back(&out_color_polys, out_color_polys.size(), color_poly.size(), res.colors[k]);
                color_poly.emplace_back(res.points[k]);
            }
        }
        out_color_polys.push_back(color_poly);
    }
}


void MultiMaterialSegmentation::coloredLineSegmentMatching2(Shape& polys, Shape& color_line_polys, Shape& out_color_polys, std::vector<Segment>& out_color_segments)
{
    for (int i = 0; i < polys.size(); ++i)
    {
        if (polys[i].front() != polys[i].back()) {
            polys[i].push_back(polys[i].front());
        }
    }

    auto copyPolygons = [&polys, &out_color_polys, &out_color_segments](int color){
        for (int i = 0; i < polys.size(); ++i)
        {
            Polygon color_poly;
            for (int j = 0; j < polys[i].size(); ++j)
            {
                color_poly.push_back(polys[i][j]);
                out_color_segments.emplace_back(&out_color_polys, i, j, color);
            }
            out_color_polys.push_back(color_poly);
        }
    };

    Shape diffLines = polys.difference(color_line_polys);


    if (diffLines.empty()) {
        copyPolygons(MESH_PAINTING_COLOR);
        return;
    }

    Shape interLines = polys.intersection(color_line_polys);

    if (interLines.empty()) {
        copyPolygons(MESH_NO_PAINTING_COLOR);
        return;
    }

    int color_size = diffLines.size();

//    diffLines.print();
//    interLines.print();

    diffLines.push_back(interLines);

    auto getColor = [&color_size](int size){
        return size < color_size ? MESH_NO_PAINTING_COLOR : MESH_PAINTING_COLOR;
    };

    int use_count = 0;
    std::vector<bool> used(diffLines.size(), false);

    // The lines front in the first and back in the second;
    std::unordered_map<Point2LL, std::vector<int>> point_lines_map;
    std::vector<std::vector<int>> color_segments_tmp;

    for (int i = 0; i < diffLines.size(); ++i)
    {
        Point2LL& front = diffLines[i].front();
        Point2LL& back = diffLines[i].back();
        if (front == back) {
            use_count++;
            used[i] = true;
            out_color_polys.push_back(diffLines[i]);
            color_segments_tmp.emplace_back();
            color_segments_tmp.back().resize(diffLines[i].size(), getColor(i));
            continue;
        }
        if (point_lines_map.find(front) == point_lines_map.end()) {
            point_lines_map[front] = std::vector<int>();
        }
        if (point_lines_map.find(back) == point_lines_map.end()) {
            point_lines_map[back] = std::vector<int>();
        }

        point_lines_map[front].push_back(i);
        point_lines_map[back].push_back(i);
    }

    while (use_count < diffLines.size()) {
        Polygon poly;
        color_segments_tmp.emplace_back();
        for (int i = 0; i < diffLines.size(); ++i)
        {
            if (used[i]) {
                continue;
            }
            used[i] = true;
            use_count++;
            for (int j = 0; j < diffLines[i].size(); ++j)
            {
                poly.push_back(diffLines[i][j]);
                color_segments_tmp.back().emplace_back(getColor(i));
            }
            break;
        }

        while (use_count < diffLines.size()) {
            auto next = point_lines_map.find(poly.back());
            assert(next != point_lines_map.end());
            if (next == point_lines_map.end()) {
                spdlog::warn("Abnormal color matching of line segments, next is null");
                break;
            }
            int next_idx = -1;
            for (int i = 0; i < next->second.size(); ++i)
            {
                if (used[next->second[i]]) {
                    continue;
                }
                next_idx = next->second[i];
            }
            if (next_idx == -1) {
                spdlog::warn("Abnormal color matching of line segments, next_idx is -1");
                break;
            }
            use_count++;
            used[next_idx] = true;
            auto next_poly = diffLines[next_idx];
            bool is_first = poly.back() == next_poly.front();
            for (int j = is_first ? 0 : next_poly.size() - 1; is_first ? j <= next_poly.size() - 1: j >= 0;is_first ? ++j:--j)
            {
                poly.push_back(next_poly[j]);
                color_segments_tmp.back().emplace_back(getColor(next_idx));
            }
            if (poly.front() == poly.back()) {
                break;
            }
        }
        out_color_polys.push_back(poly);
    }

    assert(polys.size() == out_color_polys.size());
    assert(out_color_polys.size() == color_segments_tmp.size());

    // Check the direction of polygons after splicing
    std::vector<int> areas;
    for (int i = 0; i < polys.size(); ++i)
    {
        areas.push_back(polys[i].area());
    }
    for (int i = 0; i < out_color_polys.size(); ++i)
    {
        int area = out_color_polys[i].area();
        int abs_area = std::abs(area);
        int min_area = std::abs(areas[0]) - abs_area;
        int min_area_idx = 0;
        for (int j = 0; j < areas.size(); ++j)
        {
            int area_tmp = std::abs(areas[j]) - std::abs(area);
            if (area_tmp < min_area) {
                min_area = area_tmp;
                min_area_idx = j;
            }
        }
        if ((bool)(area > 0) != (bool)(areas[min_area_idx] > 0)) {
            out_color_polys[i].reverse();
            std::reverse(color_segments_tmp[i].begin(), color_segments_tmp[i].end());
        }
    }

    // Check edge redundant small line segments
    coord_t min_offset_s_len2 = MM2INT(0.06) * MM2INT(0.06);
    for (int i = 0; i < out_color_polys.size(); ++i)
    {
        for (int j = 0; j < out_color_polys[i].size(); ++j)
        {
            if (color_segments_tmp[i][j] != MESH_PAINTING_COLOR) {
                continue ;
            }
            int prev_color = color_segments_tmp[i][(j - 1) % out_color_polys[i].size()];
            int next_color = color_segments_tmp[i][(j + 1) % out_color_polys[i].size()];
            if (prev_color == next_color) {
                continue ;
            }
            coord_t len2 = vSize2((out_color_polys[i][(j + 1) % out_color_polys[i].size()] - out_color_polys[i][j]));
            if (len2 > min_offset_s_len2) {
                continue ;
            }
            color_segments_tmp[i][j] = MESH_NO_PAINTING_COLOR;
        }
    }

    for (int i = 0; i < color_segments_tmp.size(); ++i)
    {
        for (int j = 0; j < color_segments_tmp[i].size(); ++j)
        {
            out_color_segments.emplace_back(&out_color_polys, i, j, color_segments_tmp[i][j]);
        }
    }

}

MultiMaterialSegmentation::Line MultiMaterialSegmentation::linePolygonsIntersection(Point2LL& p1, Point2LL& p2, Shape& line_polys)
{
    Shape polys;
    Polygon poly;
    poly.push_back(p1);
    poly.push_back(p2);
    polys.push_back(poly);
    Shape res = line_polys.intersection(polys);

    Line line;
    if (res.size() == 0)
    {
        line.points.emplace_back(p1);
        line.points.emplace_back(p2);
        line.colors.emplace_back(MESH_NO_PAINTING_COLOR);
        return line;
    }
    coord_t len2 = MM2INT(0.05) * MM2INT(0.05);
    if (res.size() == 1)
    {
        Point2LL& res_p1 = res[0][0];
        Point2LL& res_p2 = res[0][1];
        if ((p1 == res_p1 && p2 == res_p2) || (p1 == res_p2 && p2 == res_p1))
        {
            line.points.emplace_back(p1);
            line.points.emplace_back(p2);
            line.colors.emplace_back(MESH_PAINTING_COLOR);
            return line;
        }
        if (p1 == res_p1 || p1 == res_p2)
        {
            Point2LL middle = p1 == res_p1 ? res_p2 : res_p1;
            if (vSize2(middle - p1) < len2)
            {
                line.points.emplace_back(p1);
                line.points.emplace_back(p2);
                line.colors.emplace_back(MESH_NO_PAINTING_COLOR);
            }
            else
            {
                line.points.emplace_back(p1);
                line.points.emplace_back(middle);
                line.points.emplace_back(p2);
                line.colors.emplace_back(MESH_PAINTING_COLOR);
                line.colors.emplace_back(MESH_NO_PAINTING_COLOR);
            }
            return line;
        }
        if (p2 == res_p1 || p2 == res_p2)
        {
            Point2LL middle = p2 == res_p1 ? res_p2 : res_p1;
            if (vSize2(p2 - middle) < len2)
            {
                line.points.emplace_back(p1);
                line.points.emplace_back(p2);
                line.colors.emplace_back(MESH_NO_PAINTING_COLOR);
            }
            else
            {
                line.points.emplace_back(p1);
                line.points.emplace_back(middle);
                line.points.emplace_back(p2);
                line.colors.emplace_back(MESH_NO_PAINTING_COLOR);
                line.colors.emplace_back(MESH_PAINTING_COLOR);
            }
            return line;
        }
    }

    line.points.emplace_back(p1);
    line.points.emplace_back(p2);

    for (int i = 0; i < res.size(); ++i)
    {
        assert(res[i].size() == 2);
        for (int j = 0; j < res[i].size(); ++j)
        {
            line.points.emplace_back(res[i][j]);
        }
    }

    bool sort_key = std::abs(p1.X - p2.X) > std::abs(p1.Y - p2.Y);

    std::sort(line.points.begin(),
              line.points.end(),
              [&sort_key](Point2LL& a, Point2LL& b)
              {
                  if (sort_key)
                  {
                      return a.X < b.X;
                  }
                  else
                  {
                      return a.Y < b.Y;
                  }
              });
    if (line.points.front() != p1)
    {
        std::reverse(line.points.begin(), line.points.end());
        assert(line.points.front() == p1);
        assert(line.points.back() == p2);
    }

    int color = MESH_NO_PAINTING_COLOR;
    for (int i = 0; i < line.points.size() - 1; ++i)
    {
        line.colors.emplace_back(color);
        color = color == MESH_NO_PAINTING_COLOR ? MESH_PAINTING_COLOR : MESH_NO_PAINTING_COLOR;
    }
    if (line.points[0] == line.points[1])
    {
        line.points.erase(line.points.begin());
        line.colors.erase(line.colors.begin());
    }
    if (line.points[line.points.size() - 2] == line.points.back())
    {
        line.points.pop_back();
        line.colors.pop_back();
    }
    return line;
}

Shape MultiMaterialSegmentation::paintingSlicerLayerColoredFaces(SlicerLayer& layer, const Mesh* p_mesh, coord_t min_z, coord_t max_z)
{
    if (layer.color_faces_idx_.empty())
    {
        return Shape();
    }
    Shape polygons;
    Point3LL z_p(0, 0, 1);

    for (int i = 0; i < layer.color_faces_idx_.size(); ++i)
    {
        int face_idx = layer.color_faces_idx_[i];
        auto& face = p_mesh->faces_[face_idx];
        Point3LL vs[3];
        vs[0] = p_mesh->vertices_[face.vertex_index_[0]].p_;
        vs[1] = p_mesh->vertices_[face.vertex_index_[1]].p_;
        vs[2] = p_mesh->vertices_[face.vertex_index_[2]].p_;

        auto cb = vs[2] - vs[1];
        auto ab = vs[0] - vs[1];
        auto normal = cb.cross(ab);

        double cos = normal.dot(z_p) / std::sqrt(normal.vSize2());
        double angle = std::acos(cos);
        if (angle > M_PI / 3 && angle < M_PI / 3 * 2)
        {
            continue;
        }

        std::sort(vs, vs + 3, [](Point3LL& p1, Point3LL& p2) { return p1.z_ < p2.z_; });
        if (vs[0].z_ > max_z || vs[2].z_ < min_z)
        {
            continue;
        }

        Polygon poly;
        if (vs[0].z_ >= min_z && vs[2].z_ <= max_z)
        {
            poly.push_back(Point2LL(vs[0].x_, vs[0].y_));
            poly.push_back(Point2LL(vs[1].x_, vs[1].y_));
            poly.push_back(Point2LL(vs[2].x_, vs[2].y_));
        }
        else if (vs[0].z_ < min_z && vs[2].z_ <= max_z)
        {
            poly.push_back(Point2LL(vs[2].x_, vs[2].y_));
            if (vs[1].z_ >= min_z)
            {
                poly.push_back(Point2LL(vs[1].x_, vs[1].y_));
                Point3LL p1 = getPoint3ByZ(vs[1], vs[0], min_z);
                Point3LL p2 = getPoint3ByZ(vs[0], vs[2], min_z);
                poly.push_back(Point2LL(p1.x_, p1.y_));
                poly.push_back(Point2LL(p2.x_, p2.y_));
            }
            else
            {
                Point3LL p1 = getPoint3ByZ(vs[2], vs[1], min_z);
                Point3LL p2 = getPoint3ByZ(vs[0], vs[2], min_z);
                poly.push_back(Point2LL(p1.x_, p1.y_));
                poly.push_back(Point2LL(p2.x_, p2.y_));
            }
        }
        else if (vs[0].z_ >= min_z && vs[2].z_ > max_z)
        {
            poly.push_back(Point2LL(vs[0].x_, vs[0].y_));
            if (vs[1].z_ <= max_z)
            {
                poly.push_back(Point2LL(vs[1].x_, vs[1].y_));
                Point3LL p1 = getPoint3ByZ(vs[1], vs[2], max_z);
                Point3LL p2 = getPoint3ByZ(vs[2], vs[0], max_z);
                poly.push_back(Point2LL(p1.x_, p1.y_));
                poly.push_back(Point2LL(p2.x_, p2.y_));
            }
            else
            {
                Point3LL p1 = getPoint3ByZ(vs[0], vs[1], max_z);
                Point3LL p2 = getPoint3ByZ(vs[2], vs[0], max_z);
                poly.push_back(Point2LL(p1.x_, p1.y_));
                poly.push_back(Point2LL(p2.x_, p2.y_));
            }
        }
        else
        {
            if (vs[1].z_ > max_z)
            {
                Point3LL p1 = getPoint3ByZ(vs[0], vs[2], max_z);
                Point3LL p2 = getPoint3ByZ(vs[1], vs[0], max_z);
                Point3LL p3 = getPoint3ByZ(vs[1], vs[0], min_z);
                Point3LL p4 = getPoint3ByZ(vs[0], vs[2], min_z);
                poly.push_back(Point2LL(p1.x_, p1.y_));
                poly.push_back(Point2LL(p2.x_, p2.y_));
                poly.push_back(Point2LL(p3.x_, p3.y_));
                poly.push_back(Point2LL(p4.x_, p4.y_));
            }
            else if (vs[1].z_ < min_z)
            {
                Point3LL p1 = getPoint3ByZ(vs[0], vs[2], max_z);
                Point3LL p2 = getPoint3ByZ(vs[2], vs[1], max_z);
                Point3LL p3 = getPoint3ByZ(vs[2], vs[1], min_z);
                Point3LL p4 = getPoint3ByZ(vs[0], vs[2], min_z);
                poly.push_back(Point2LL(p1.x_, p1.y_));
                poly.push_back(Point2LL(p2.x_, p2.y_));
                poly.push_back(Point2LL(p3.x_, p3.y_));
                poly.push_back(Point2LL(p4.x_, p4.y_));
            }
            else
            {
                Point3LL p1 = getPoint3ByZ(vs[0], vs[2], max_z);
                Point3LL p2 = getPoint3ByZ(vs[2], vs[1], max_z);
                Point3LL p3 = vs[1];
                Point3LL p4 = getPoint3ByZ(vs[1], vs[0], min_z);
                Point3LL p5 = getPoint3ByZ(vs[0], vs[2], min_z);
                poly.push_back(Point2LL(p1.x_, p1.y_));
                poly.push_back(Point2LL(p2.x_, p2.y_));
                poly.push_back(Point2LL(p3.x_, p3.y_));
                poly.push_back(Point2LL(p4.x_, p4.y_));
                poly.push_back(Point2LL(p5.x_, p5.y_));
            }
        }

        if (poly.empty())
        {
            continue;
        }

        poly.sortArea();
        polygons.push_back(poly);
    }

    polygons = polygons.unionPolygons();
    polygons = polygons.intersection(layer.polygons_);
    return polygons;
}

Point3LL MultiMaterialSegmentation::getPoint3ByZ(Point3LL& p1, Point3LL& p2, int z)
{
    if (p1.z_ == z)
    {
        return p1;
    }
    if (p2.z_ == z)
    {
        return p2;
    }
    double k = (double)(z - p1.z_) / (p2.z_ - p1.z_);
    return p1 + (p2 - p1) * k;
}

} // namespace cura
