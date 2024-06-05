#ifndef HitableLIST_H
#define HitableLIST_H

#include "hitable.h"

class HitableList : public Hitable {
 public:
  __device__ HitableList() {}
  __device__ HitableList(Hitable** l, int n) {
    list = l;
    listSize = n;
  }
  __device__ virtual bool hit(const Ray& r, float tmin, float tmax,
                              HitRecord& rec) const;
  Hitable** list;
  int listSize;
};

__device__ bool HitableList::hit(const Ray& r, float tMin, float tMax,
                                 HitRecord& rec) const {
  HitRecord tempRec;
  bool hitAnything = false;
  float closestSoFar = tMax;
  for (int i = 0; i < listSize; i++) {
    if (list[i]->hit(r, tMin, closestSoFar, tempRec)) {
      hitAnything = true;
      closestSoFar = tempRec.t;
      rec = tempRec;
    }
  }
  return hitAnything;
}

#endif