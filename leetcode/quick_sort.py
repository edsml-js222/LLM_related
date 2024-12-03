import numpy as np

def quick_sort(arr: list[int]) -> list[int]:
    """sort a list using quick_sort"""
    def partition(low, high):
        """assistant funciton"""
        pivot = arr[high] # select the last element as the pivot
        i = low - 1 # the index of element whose value is smaller than the pivot

        # move all the elements whose value is smaller than pivot to the left
        for j in range(low, high):
            if arr[j] < pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        # move pivot element to the middle
        arr[i + 1], arr[high] = arr[high], arr[i+1]
        return i + 1 # the index of the pivot element 
    
    def quick_sort_recursive(low ,high):
        if low < high:
            # get the index of pivot element
            pivot_idx = partition(low, high)
            # recursively sort the elements on the left and right side of the pivot element 
            quick_sort_recursive(low, pivot_idx - 1)
            quick_sort_recursive(pivot_idx + 1, high)
    quick_sort_recursive(0, len(arr) - 1)

if __name__ == "__main__":
    arr = np.random.randint(0, 30, 10)
    print(f"array before sort: {arr}")
    quick_sort(arr)
    print(f"array after sort: {arr}")