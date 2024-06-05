#ifndef SPRITE_RENDERER_H
#define SPRITE_RENDERER_H
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "texture.hpp"
#include "shader.hpp"


class SpriteRenderer
{
public:
    SpriteRenderer()=default;
    SpriteRenderer(const Shader &shader);
    ~SpriteRenderer();
    void drawSprite(const Texture2D &texture, glm::vec2 position, glm::vec2 size = glm::vec2(10.0f, 10.0f), float rotate = 0.0f, glm::vec3 color = glm::vec3(1.0f));
private:
    Shader       shader; 
    unsigned int quadVAO;
    void initRenderData();
};
#endif