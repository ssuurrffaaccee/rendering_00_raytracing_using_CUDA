#ifndef MEMORY_H
#define MEMORY_H
template <typename T>
class MemoryRecorder {
 public:
  // __device__ static MemoryRecorder<T>& getIntance() {
  //   __device__ static MemoryRecorder<T> instance_{};
  //   return instance_;
  // }
  __device__ MemoryRecorder() {
    int pre_num = 50000;
    data_ = (T**)malloc(pre_num * sizeof(T*));
    cap_ = pre_num;
    size_ = 0;
  }
  __device__ ~MemoryRecorder() {
    for (int i = 0; i < size_; i++) {
      delete data_[i];
    }
    free(data_);
  }
  template <typename M>
  __device__ void record(M* ptr) {
    if (size_ == cap_) {
      expand();
    }
    data_[size_] = (T*)ptr;
    size_++;
  }

 private:
  T** data_;
  int cap_;
  int size_;
  __device__ void expand() {
    int new_cap = 2 * cap_;
    T** old_data = data_;
    data_ = (T**)malloc(new_cap * sizeof(T*));
    cap_ = new_cap;
    for (int i = 0; i < size_; i++) {
      data_[i] = old_data[i];
    }
    free(old_data);
  }

};
#endif