def solution(nums: list) -> int:
    length = len(nums)
    if length == 1:
        return 1
    i = 1
    while i < length:
        if nums[i] == nums[i - 1]:
            nums.pop(i)
            length -= 1
        else:
            i += 1
    return i


if __name__ == '__main__':
    nums = [2, 2, 4, 5, 6, 6]
    solution(nums)
    print(nums)
