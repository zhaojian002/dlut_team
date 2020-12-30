# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""
import argparse
import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw, ImageTk

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

import blue_1
import day10
import tkinter as tk
from tkinter import filedialog

#修改：用来记录车辆位置并记录的变量

left_re=0
top_re=0
flag_num=0
bottom_re=0
right_re=0
flag_count=0
count_num=0
flag_count1=0
count_num1=0
over_speed_num = 0
car_num = 0
car_counttime = 0

license_num = ["","","","",""]

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label1 = label[0:3]
            if label1 != "tra" and label1 != "per" and label1!="umb" and label1!="bic" and label1!="tru" and \
                    label1!="han" and label1!="fir" and label1!="pot" and label1!="bus":
                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
               # print(label, (left, top), (right, bottom))

                # 修改：用来记录车辆位置并记录
                global left_re

                global top_re

                global flag_num

                global bottom_re

                global right_re

                global car_counttime

                global car_num


                if (left_re != 0 and top_re != 0) and (
                        (top_re - top > 20 and top_re - top < 40) or (top_re - top < -20 and top_re - top > -40)) and (
                        left<720 and left>520) and(
                        top > 600 and top < 675) and flag_num < 1:

                    top_re = top
                    right_re=right
                    left_re=left
                    bottom_re=bottom
                    flag_num = flag_num + 1

                if((left<578  and left>520) and (top>600 and top<675)) and flag_num==0:
                    left_re = left
                    top_re = top

                if left > 520 and (
                        top > 600 and top < 640) and car_counttime<=1:
                    car_num = car_num +1
                    car_counttime = 1

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw
        print("**************"+str(car_num))
        if car_counttime==1:
            car_counttime = car_counttime +2
        if car_counttime>1:
            car_counttime = car_counttime + 1
        if car_counttime>=11:
            car_counttime=0
        end = timer()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    # 更改视频2
    window = tk.Tk()
    window.title('正在识别中')
    window.geometry('1900x750')
    canvas = tk.Canvas(window, width=1400, height=788, bg="white")
    canvas.place(x=20, y=90)
    canvas1 = tk.Canvas(window, width=600, height=16, bg="white")
    canvas1.place(x=2, y=2)

    vid = cv2.VideoCapture(video_path)
    #vid = cv2.VideoCapture("video-02.mp4")
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    fps_total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_pos =  0
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    output_path="video-01.mp4";
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter("video-01.mp4", video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        cv2.rectangle(frame, (520, 600), (720,675), (255, 0, 0))
        cv2.putText(frame, "check_area", (520, 600), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2, 8, 0)
        image = Image.fromarray(frame)




        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1

        #修改截取保存图片1
        global left_re

        global top_re

        global flag_num

        global bottom_re

        global right_re

        global flag_count

        global count_num

        global flag_count1

        global count_num1

        global license_num

        global over_speed_num

        # 更改视频1

        pilImage = image
        pilImage = pilImage.resize((1500, 600), Image.ANTIALIAS)

        tkImage = ImageTk.PhotoImage(image=pilImage)
        canvas.create_image(0, 0, anchor='nw', image=tkImage)
        window.update_idletasks()
        window.update()



        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        #cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        #cv2.imshow("result", result)
        # 修改截取保存图片1
        if flag_num == 1:
            print("top_re:", top_re, "bottom_re", bottom_re, "left_re:", left_re, "right_re:", right_re)
            cutting = result[top_re:bottom_re, left_re:right_re]
            cv2.imwrite('cut'+str(over_speed_num)+'.jpg', cutting)
            flag_num = 10
            flag_count = 1
        if flag_count==1:
            count_num = count_num + 1
        if count_num==5:
            blue_1.check_license_plate(over_speed_num)
            flag_count1 = 1
        if flag_count1==1:
            count_num1 = count_num1+1
        if count_num1 == 5:
            license_num[over_speed_num] = day10.distinguish_license_plate()
            left_re = 0
            top_re = 0
            flag_num = 0
            bottom_re = 0
            right_re = 0
            flag_count = 0
            count_num = 0
            flag_count1 = 0
            count_num1 = 0
            over_speed_num = over_speed_num + 1



        if isOutput:
            out.write(result)
            fps_pos = fps_pos + 1
            fill_line = canvas1.create_rectangle(1.5, 1.5, 0, 23, width=0, fill="green")
            raise_data = 600 * (fps_pos / fps_total)
            canvas1.coords(fill_line, (0, 0, raise_data, 60))
            window.update()
            if fps_pos == fps_total - 1:
                window.destroy()
                break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()





def dis_finish():
    import cv2
    # 更改视频2
    window = tk.Tk()
    window.title('成果展示')
    window.geometry('1900x750')
    canvas = tk.Canvas(window, width=1000, height=562, bg="white")
    canvas.place(x=20, y=20)
    global license_num
    global over_speed_num
    global car_num
    i=0
    label_text = tk.Label(window, text="下面车道总流量："+str(car_num))
    label_text.place(x=1020, y=5)
    while True:
        img = Image.open('cut'+str(i)+'.jpg')  # 打开图片
        photo = ImageTk.PhotoImage(img)
        label_img = tk.Label(window, image=photo)
        label_img.place(x=1020, y=155+500*i)
        license_num[i] = '超速车牌号：' + license_num[i]
        label_text = tk.Label(window, text=license_num[i].split('.', 1)[0])
        label_text.place(x=1020, y=55+500*i)
        label_text1 = tk.Label(window, text=license_num[i].split('.', 1)[1])
        label_text1.place(x=1020, y=105+500*i)
        i = i+1
        if i>over_speed_num-1:
            break
    vid = cv2.VideoCapture("video-01.mp4")
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1

        # 更改视频1

        pilImage = image
        pilImage = pilImage.resize((1000, 562), Image.ANTIALIAS)

        tkImage = ImageTk.PhotoImage(image=pilImage)
        canvas.create_image(0, 0, anchor='nw', image=tkImage)
        window.update_idletasks()
        window.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def hit_start():
    window.destroy()
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        "--input", nargs='?', type=str, required=False, default='./path2your_video',
        help="Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help="[Optional] Video output path"
    )
    FLAGS = parser.parse_args()

    detect_video(YOLO(**vars(FLAGS)), "video-02.mp4", '')
    dis_finish()
def hit_start_1():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        "--input", nargs='?', type=str, required=False, default='./path2your_video',
        help="Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help="[Optional] Video output path"
    )
    FLAGS = parser.parse_args()
    f = tk.filedialog.askopenfilename()
    for i in range(len(f)):
        if f[i]=='/':
            f = f[:i]+'\\'+f[i+1:]
    print(f)
    window.destroy()
    detect_video(YOLO(**vars(FLAGS)), f, '')

    dis_finish()
if __name__ == '__main__':

    window = tk.Tk()
    window.title('车辆检测')
    window.geometry('500x300')
    b = tk.Button(window, text='固有视频测试', font=('Arial', 12), width=10, height=1, command=hit_start)
    b.pack()
    b1 = tk.Button(window, text='本地视频检测', font=('Arial', 12), width=10, height=1, command=hit_start_1)
    b1.pack()
    window.mainloop()
