#pragma once
#include <cstddef>
#include <memory>
#include "./Helper.hpp"

#define inline_buffer_size 256

namespace MyVector{

template <typename T> class buffer {
 private:
  T* ptr_;
  std::size_t size_;
  std::size_t capacity_;

 protected:
  buffer(std::size_t sz) noexcept : size_(sz), capacity_(sz) {}

  buffer(T* p = nullptr, std::size_t sz = 0, std::size_t cap = 0) noexcept
      : ptr_(p),
        size_(sz),
        capacity_(cap) {}

  void set(T* buf_data, std::size_t buf_capacity) noexcept {
    ptr_ = buf_data;
    capacity_ = buf_capacity;
  }

  virtual void grow(std::size_t capacity) = 0;

 public:
  using value_type = T;
  using const_reference = const T&;

  buffer(const buffer&) = delete;
  void operator=(const buffer&) = delete;
  virtual ~buffer() = default;

  T* begin() noexcept { return ptr_; }
  T* end() noexcept { return ptr_ + size_; }

  std::size_t size() const noexcept { return size_; }

  std::size_t capacity() const noexcept { return capacity_; }

  T* data() noexcept { return ptr_; }

  const T* data() const noexcept { return ptr_; }

  void resize(std::size_t new_size) {
    reserve(new_size);
    size_ = new_size;
  }

  void clear() { size_ = 0; }

  void reserve(std::size_t new_capacity) {
    if (new_capacity > capacity_) grow(new_capacity);
  }

  void push_back(const T& value) {
    reserve(size_ + 1);
    ptr_[size_++] = value;
  }

  template <typename U> void append(const U* begin, const U* end){
    std::size_t new_size = size_ + to_unsigned(end - begin);
    reserve(new_size);
    std::uninitialized_copy(begin, end, ptr_ + size_);
    size_ = new_size;
  };

  T& operator[](std::size_t index) { return ptr_[index]; }
  const T& operator[](std::size_t index) const { return ptr_[index]; }
};

template <typename T, std::size_t SIZE = inline_buffer_size,
          typename Allocator = std::allocator<T>>
class basic_memory_buffer : private Allocator, public buffer<T> {
 private:
  T store_[SIZE];

  void deallocate() {
    T* data = this->data();
    if (data != store_) Allocator::deallocate(data, this->capacity());
  }

 protected:
  void grow(std::size_t size) override {
    std::size_t old_capacity = this->capacity();
    // 1.5倍动态扩容
    std::size_t new_capacity = old_capacity + old_capacity / 2;
    if (size > new_capacity) new_capacity = size;
    T* old_data = this->data();
    T* new_data = std::allocator_traits<Allocator>::allocate(*this, new_capacity);
    std::uninitialized_copy(old_data, old_data + this->size(), new_data);
    this->set(new_data, new_capacity);
    if (old_data != store_) Allocator::deallocate(old_data, old_capacity);
  };

 public:
  using value_type = T;
  using const_reference = const T&;

  explicit basic_memory_buffer(const Allocator& alloc = Allocator())
      : Allocator(alloc) {
    this->set(store_, SIZE);
  }
  ~basic_memory_buffer() override { deallocate(); }

 private:
  void move(basic_memory_buffer& other) {
    Allocator &this_alloc = *this, &other_alloc = other;
    this_alloc = std::move(other_alloc);
    T* data = other.data();
    std::size_t size = other.size(), capacity = other.capacity();
    if (data == other.store_) {
      this->set(store_, capacity);
      std::uninitialized_copy(other.store_, other.store_ + size, store_);
    } else {
      this->set(data, capacity);
    //   deallocate的时候并不会真的释放other这块内存
      other.set(other.store_, 0);
    }
    this->resize(size);
  }

 public:

  basic_memory_buffer(basic_memory_buffer&& other) noexcept { move(other); }

  basic_memory_buffer& operator=(basic_memory_buffer&& other) noexcept {
    MEM_ASSERT(this != &other, "");
    deallocate();
    move(other);
    return *this;
  }

  Allocator get_allocator() const { return *this; }
};

} // end of namespace

template <typename T, std::size_t SIZE = inline_buffer_size,
          typename Allocator = std::allocator<T>>
using Vector = MyVector::basic_memory_buffer<T, SIZE, Allocator>;