def DO_STUFF(x):
  import math
  import time
  list1 = []
  for i in range(10):
    list1.append(i)
  a = x
  b = a + 1
  if x == 1:
      print("one")
  elif x == 2:
      print("two")
  else:
      pass
  
  while True:
      if a > 5:
          break
      a = a + 1
      
  return a

print(DO_STUFF(3))