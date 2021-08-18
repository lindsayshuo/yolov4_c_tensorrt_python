
import cv2
import TRTYolov4 as t

engine = t.create('./build/yolov4.engine')

img = cv2.imread('./samples/lius.jpg')

#b = t.detect(engine, img, 0.45)

b = t.detect(engine, img, 0.45)
# t.destroy(engine)
#print(engine)
print('sssss')
# print(b)


# import ctypes
#
# # ctypes.CDLL("./build/libmyplugins.so")
# so = ctypes.CDLL("../build/libyolov5_trt.so")
# engine = so.create('../yolov5s.engine')





"""
import ctypes


ctypes.CDLL("./build/libmyplugins.so")
so = ctypes.CDLL("./build/libyolov4_trt.so")

s = so.add(1, 2)

print(s)

engine = so.yolov4_trt_create(ctypes.c_char_p(bytes('yolov4.engine', 'utf-8')))
"""
