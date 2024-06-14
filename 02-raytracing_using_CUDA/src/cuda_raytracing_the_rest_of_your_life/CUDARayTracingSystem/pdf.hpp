#ifndef PDF_H
#define PDF_H
#include "hittable.hpp"
#include "onb.hpp"
#include "random.hpp"

enum class PDFType{
  UNKNOW,
  UNIFORM,
  COSINE,
  HITTABLE,
  MIXTURE,
};

class PDF {
 public:
  using Sample = Vec3;
  __device__ virtual Sample sample(curandState *local_rand_state) const {
    return Sample{};
  };
  __device__ virtual float pdf(const Sample &x,
                                curandState *local_rand_state) const {
    return 1.0f;
  };
};

class UniformSpherePDF : public PDF {
 public:
  __device__ UniformSpherePDF() {}

  __device__ float pdf(const Sample &direction,
                        curandState *local_rand_state) const override {
    return 1 / (4 * PI);
  }

  __device__ Sample sample(curandState *local_rand_state) const override {
    return random_unit_vector(local_rand_state);
  }
};

class CosineHemispherePDF : public PDF {
 public:
 __device__ CosineHemispherePDF(){
 }
  __device__ CosineHemispherePDF(const Vec3 &norm) : uvw_{} {
    uvw_.build_from_w(norm);
  }

  __device__ float pdf(const Sample &direction,
                        curandState *local_rand_state) const override {
    auto cosine_theta = dot(unit_vector(direction), uvw_.w());
    return mymax(0, cosine_theta / PI);
  }

  __device__ Sample sample(curandState *local_rand_state) const override {
    return uvw_.local(random_cosine_direction(local_rand_state));
  }

 private:
  OrthonormalBasis uvw_;
};

class HittablePDF : public PDF {
 public:
  __device__ HittablePDF(){
  }
  __device__ HittablePDF(Hittable *object, const Point3 &origin)
      : object_{object}, origin_{origin} {
      }

  __device__ float pdf(const Vec3 &direction,
                        curandState *local_rand_state) const override {
    return object_->pdf_from(origin_, direction, local_rand_state);
  }

  __device__ Sample sample(curandState *local_rand_state) const override {
    return object_->sample_from(origin_, local_rand_state);
  }

//  private:
  Hittable *object_;
  Point3 origin_;
};
class MixturePDF : public PDF {
 public:
 __device__ MixturePDF(){
 }
  __device__ MixturePDF(PDF* p0, PDF* p1)
      : p0_{p0}, p1_{p1} {
      }

  __device__ float pdf(const Vec3 &direction,
                        curandState *local_rand_state) const override {
    return 0.5 * p0_->pdf(direction, local_rand_state) +
           0.5 * p1_->pdf(direction, local_rand_state);
  }

  __device__ Sample sample(curandState *local_rand_state) const override {
    if (random_float(local_rand_state) < 0.5) {
      return p0_->sample(local_rand_state);
    } else {
      return p1_->sample(local_rand_state);
    }
  }

 private:
  PDF *p0_;
  PDF *p1_;
};

// struct DynamicPDF{
//    __device__ DynamicPDF():type_{PDFType::UNKNOW},pdf_{PDF{}}{
//    }
//    PDFType type_{PDFType::UNKNOW};
//    union PDFUnion{
//         PDF empty_pdf;
//         UniformSpherePDF uniform_pdf;
//         CosineHemispherePDF cosine_pdf;
//         HittablePDF hittable_pdf;
//         MixturePDF mixture_pdf;
//    };
//    PDFUnion pdf_;
// };

struct DynamicPDF{
   __device__ DynamicPDF():type_{PDFType::UNKNOW}{
   }
   PDFType type_{PDFType::UNKNOW};
   PDF empty_pdf;
   UniformSpherePDF uniform_pdf;
   CosineHemispherePDF cosine_pdf;
   HittablePDF hittable_pdf;
   MixturePDF mixture_pdf;
};

#endif