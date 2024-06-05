class Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z
    def __str__(self) -> str:
        return "Vec3(%f,%f,%f)"%(self.x,self.y,self.z)


def diff3(x: Vec3, y: Vec3):
    return Vec3(x.x - y.x, x.y - y.y, x.z - y.z)


def length(x: Vec3):
    import math

    return math.sqrt(x.x * x.x + x.y * x.y + x.z * x.z)


def normalize(x: Vec3):
    l = length(x)
    return Vec3(x.x / l, x.y / l, x.z / l)


def build_v3(values):
    return Vec3(values[0],values[1],values[2])


class Vec2:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y
    def __str__(self) -> str:
        return "Vec2(%f,%f)"%(self.x,self.y)
def build_v2(values):
    return Vec2(values[0],values[1])

def diff2(x: Vec2, y: Vec2):
    return Vec2(x.x - y.x, x.y - y.y)

def flatten(pos: Vec3, norm: Vec3, uv: Vec2, tangent: Vec3, bitangent: Vec3):
    return (
        pos.x,
        pos.y,
        pos.z,
        norm.x,
        norm.y,
        norm.z,
        uv.x,
        uv.y,
        tangent.x,
        tangent.y,
        tangent.z,
        bitangent.x,
        bitangent.y,
        bitangent.z,
    )


def triangle_tangent_cal(
    pos1: Vec3, pos2: Vec3, pos3: Vec3, uv1: Vec2, uv2: Vec2, uv3: Vec2, norm: Vec3
):
    edge1 = diff3(pos2,pos1)
    edge2 = diff3(pos3,pos1)
    deltaUV1 = diff2(uv2,uv1)
    deltaUV2 = diff2(uv3,uv1)
    tangent = Vec3()
    bitangent = Vec3()
    f = 1.0 / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y)
    tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x)
    tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y)
    tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z)

    bitangent.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x)
    bitangent.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y)
    bitangent.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z)
    return [
        flatten(pos1, norm, uv1, normalize(tangent), normalize(bitangent)),
        flatten(pos2, norm, uv2, normalize(tangent), normalize(bitangent)),
        flatten(pos3, norm, uv3, normalize(tangent), normalize(bitangent)),
    ]
cube =[
     # back face
      -1.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0,  # bottom-left
      1.0, 1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 1.0,    # top-right
      1.0, -1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0,   # bottom-right
      1.0, 1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 1.0,    # top-right
      -1.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0,  # bottom-left
      -1.0, 1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 1.0,   # top-left
      # front face
      -1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,  # bottom-left
      1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,   # bottom-right
      1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,    # top-right
      1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,    # top-right
      -1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,   # top-left
      -1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,  # bottom-left
      # left face
      -1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0,    # top-right
      -1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0,   # top-left
      -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0,  # bottom-left
      -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0,  # bottom-left
      -1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0,   # bottom-right
      -1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0,    # top-right
                                                           # right face
      1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,      # top-left
      1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0,    # bottom-right
      1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 1.0,     # top-right
      1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0,    # bottom-right
      1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,      # top-left
      1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,     # bottom-left
      # bottom face
      -1.0, -1.0, -1.0, 0.0, -1.0, 0.0, 0.0, 1.0,  # top-right
      1.0, -1.0, -1.0, 0.0, -1.0, 0.0, 1.0, 1.0,   # top-left
      1.0, -1.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0,    # bottom-left
      1.0, -1.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0,    # bottom-left
      -1.0, -1.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0,   # bottom-right
      -1.0, -1.0, -1.0, 0.0, -1.0, 0.0, 0.0, 1.0,  # top-right
      # top face
      -1.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 1.0,  # top-left
      1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,    # bottom-right
      1.0, 1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 1.0,   # top-right
      1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,    # bottom-right
      -1.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 1.0,  # top-left
      -1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0    # bottom-left
]
point_len = 3 + 3 + 2
triangle_len = point_len * 3
print(len(cube) / triangle_len)
for index in range(int(len(cube) /triangle_len)):
    triangle = cube[index*triangle_len: (index+1)*triangle_len]
    point1 = triangle[:point_len]
    point2 = triangle[point_len : point_len * 2]
    point3 = triangle[point_len * 2 :]
    pos1 = build_v3(point1[:3])
    pos2 = build_v3(point2[:3])
    pos3 = build_v3(point3[:3])
    norm = build_v3(point1[3:6])
    uv1 = build_v2(point1[6:8])
    uv2 = build_v2(point2[6:8])
    uv3 = build_v2(point3[6:8])
    res = triangle_tangent_cal(pos1, pos2, pos3, uv1, uv2, uv3, norm)
    print()
    for r in res:
        print(str(r)[1:-1]+',')


# # positions
#         glm::vec3 pos1(-1.0,  1.0, 0.0f);
#         glm::vec3 pos2(-1.0, -1.0, 0.0f);
#         glm::vec3 pos3( 1.0, -1.0, 0.0f);
#         glm::vec3 pos4( 1.0,  1.0, 0.0f);
#         # texture coordinates
#         glm::vec2 uv1(0.0, 1.0f);
#         glm::vec2 uv2(0.0, 0.0f);
#         glm::vec2 uv3(1.0, 0.0f);
#         glm::vec2 uv4(1.0, 1.0f);
#         # normal vector
#         glm::vec3 nm(0.0, 0.0, 1.0f);

#         # calculate tangent/bitangent vectors of both triangles
#         glm::vec3 tangent1, bitangent1;
#         glm::vec3 tangent2, bitangent2;
#         # triangle 1
#         # ----------
#         glm::vec3 edge1 = pos2 - pos1;
#         glm::vec3 edge2 = pos3 - pos1;
#         glm::vec2 deltaUV1 = uv2 - uv1;
#         glm::vec2 deltaUV2 = uv3 - uv1;

#         float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

#         tangent1.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
#         tangent1.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
#         tangent1.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);

#         bitangent1.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
#         bitangent1.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
#         bitangent1.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);

#         # triangle 2
#         # ----------
#         edge1 = pos3 - pos1;
#         edge2 = pos4 - pos1;
#         deltaUV1 = uv3 - uv1;
#         deltaUV2 = uv4 - uv1;

#         f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

#         tangent2.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
#         tangent2.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
#         tangent2.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);


#         bitangent2.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
#         bitangent2.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
#         bitangent2.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);


#         float quadVertices[] = {
#             # positions            # normal         # texcoords  # tangent                          # bitangent
#             pos1.x, pos1.y, pos1.z, nm.x, nm.y, nm.z, uv1.x, uv1.y, tangent1.x, tangent1.y, tangent1.z, bitangent1.x, bitangent1.y, bitangent1.z,
#             pos2.x, pos2.y, pos2.z, nm.x, nm.y, nm.z, uv2.x, uv2.y, tangent1.x, tangent1.y, tangent1.z, bitangent1.x, bitangent1.y, bitangent1.z,
#             pos3.x, pos3.y, pos3.z, nm.x, nm.y, nm.z, uv3.x, uv3.y, tangent1.x, tangent1.y, tangent1.z, bitangent1.x, bitangent1.y, bitangent1.z,

#             pos1.x, pos1.y, pos1.z, nm.x, nm.y, nm.z, uv1.x, uv1.y, tangent2.x, tangent2.y, tangent2.z, bitangent2.x, bitangent2.y, bitangent2.z,
#             pos3.x, pos3.y, pos3.z, nm.x, nm.y, nm.z, uv3.x, uv3.y, tangent2.x, tangent2.y, tangent2.z, bitangent2.x, bitangent2.y, bitangent2.z,
#             pos4.x, pos4.y, pos4.z, nm.x, nm.y, nm.z, uv4.x, uv4.y, tangent2.x, tangent2.y, tangent2.z, bitangent2.x, bitangent2.y, bitangent2.z
#         };
