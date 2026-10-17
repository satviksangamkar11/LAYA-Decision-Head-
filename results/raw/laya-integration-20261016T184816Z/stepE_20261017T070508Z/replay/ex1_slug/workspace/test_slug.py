from slug import slugify

assert slugify("Hello, World!") == "hello-world", slugify("Hello, World!")
assert slugify("  A--B  ") == "a-b", slugify("  A--B  ")
assert slugify("") == ""
print("all tests passed")
