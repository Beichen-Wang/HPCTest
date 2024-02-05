#pragma once

#include <vector>
#include <atomic>
#include <memory>
#include <functional>
#include <bitset>
#include "SpinLock.hpp"

template <typename T, int SIZE = 4>
class ObjectPool
{
public:
    using type = T;
    using reference = T &;
    using const_reference = const T &;
    using pointer = T *;
    using const_pointer = const T *;
    using shared_pointer = std::shared_ptr<T>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    constexpr static size_type type_size = sizeof(T);
    constexpr static size_type FixedSize = 512;

protected:
    std::unique_ptr<T[]> buffer_{nullptr};
    size_type capacity_{0};
    std::bitset<FixedSize> bitmap_;
    spinlock sp_;

    size_t find_first() const noexcept
    {
        size_t index = 0;
        for(; index < capacity_; ++index)
        {
            if(bitmap_.test(index))
            {
                return index;
            }
        }
        return index;
    }

    pointer allocate() noexcept
    {
        spinlock::scoped_lock lock_guard(sp_);
        if (bitmap_.count() == 0)
        {
            return nullptr;
        }
        auto index = find_first();
        if (index >= capacity_)
        {
            return nullptr;
        }
        bitmap_.reset(index);
        return buffer_.get() + index;
    }

    void deallocate(pointer p) noexcept
    {
        spinlock::scoped_lock lock_guard(sp_);
        difference_type index = p - buffer_.get();
        if (index < 0 || index >= static_cast<difference_type>(capacity_))
        {
            return;
        }
        bitmap_.set(index);
    }

    bool is_from_pool(pointer p) const noexcept
    {
        return p >= buffer_.get() && p < buffer_.get() + capacity_;
    }

public:
    ObjectPool() {
        std::cout << "construct objectpool" << std::endl;
        init(SIZE);
    };

    ObjectPool(const ObjectPool &) = delete;
    ObjectPool &operator=(const ObjectPool &) = delete;

    ObjectPool(ObjectPool &&) = delete;
    ObjectPool &operator=(ObjectPool &&) = delete;

    ~ObjectPool() = default;

    void init(size_type n)
    {
        try {
            buffer_ = std::make_unique<T[]>(n);
        }
        catch (std::bad_alloc &e)
        {
            buffer_ = nullptr;
            capacity_ = 0;
            return;
        }
        
        capacity_ = n;
        bitmap_.reset();

        for (size_type i = 0; i < n; ++i)
        {
            bitmap_.set(i);
        }
    }

    size_type size()  noexcept
    {
        spinlock::scoped_lock lock_guard(sp_);
        return bitmap_.count();
    }

    size_type capacity() const noexcept
    {
        return capacity_;
    }

    bool empty()  noexcept
    {
        spinlock::scoped_lock lock_guard(sp_);
        return bitmap_.none();
    }

    bool full()  noexcept
    {
        spinlock::scoped_lock lock_guard(sp_);
        return bitmap_.count() == capacity_;
    }

    shared_pointer get_shared_pointer()
    {
        auto p = allocate();
        return shared_pointer(p, [this](pointer p)
                                {
        if(p != nullptr){
            deallocate(p);
        } });
    }
};

