from clamp import clamp

assert clamp(5, 0, 10) == 5, clamp(5, 0, 10)
assert clamp(-3, 0, 10) == 0, clamp(-3, 0, 10)
assert clamp(99, 0, 10) == 10, clamp(99, 0, 10)
print("all tests passed")
