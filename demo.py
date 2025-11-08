import cv2
import numpy as np
from PIL import ImageTk
from model import image_to_tensor, deepnn
import tensorflow._api.v2.compat.v1 as tf
import tensorflow as tf2
import tkinter
from tkinter import *
import PIL.Image
from tkinter import filedialog
from tkinter.messagebox import askyesno
from tkinter import messagebox
import gc
import os


CASC_PATH = './data/haarcascade_files/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
flags = tf.app.flags
flags.DEFINE_string('checkpoint_dir', './ckpt',
                    'Path to model file.')
flags.DEFINE_string('train_data', './data/fer2013/fer2013.csv',
                    'Path to training data.')
flags.DEFINE_string('valid_data', './valid_sets/',
                    'Path to training data.')
flags.DEFINE_boolean('show_box', True,
                    'If true, the results will show detection box')
FLAGS = flags.FLAGS


def format_image(image):
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # 找人脸，返回vector保存各个人脸的坐标、大小
  faces = cascade_classifier.detectMultiScale(
    image,
    scaleFactor = 1.1,
    minNeighbors = 5
  )
  # None is no face found in image
  if not len(faces) > 0:
    return None, None
  max_are_face = faces[0]
  for face in faces: # 找大小最大的，作为人脸
    if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
      max_are_face = face
  # face to image
  face_coor =  max_are_face
  image = image[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]
  # Resize image to network size
  try:
    image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
  except Exception:
    print("[+} Problem during resize")
    return None, None
  return  image, face_coor    # 图像，坐标


def resize_image(image, size):
  try:
    image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC) / 255.
  except Exception:
    print("+} Problem during resize")
    return None
  return image


def tkimage(f):
  # frame = cv2.flip(f, 1)  # 摄像头翻转
  cvimage = cv2.cvtColor(f, cv2.COLOR_BGR2RGBA)
  pilImage = PIL.Image.fromarray(cvimage)
  pilImage = pilImage.resize((550, 410), PIL.Image.ANTIALIAS)
  tkImage = ImageTk.PhotoImage(image=pilImage)
  return tkImage


# 实时视频
def demo(showBox=False):
  video_captor = cv2.VideoCapture(0)  # 获得电脑摄像
  emoji_face = []
  result = None
  num = 0
  while flag==False:
    ret, frame = video_captor.read()  # 逐帧读取视频
    detected_face, face_coor = format_image(frame)
    if showBox:
      if face_coor is not None:
        [x,y,w,h] = face_coor
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)  # 标记矩形框
    if cv2.waitKey(200):
      if detected_face is not None:
        tensor = image_to_tensor(detected_face)
        result = sess.run(probs, feed_dict={face_x: tensor})  # 代入训练网络
    if result is not None:
      for index, emotion in enumerate(EMOTIONS):
        cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
        cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4),
                      (255, 0, 0), -1)
        emoji_face = feelings_faces[np.argmax(result[0])]
      for c in range(0, 3):
        emoji_face = cv2.resize(emoji_face, (w, h))
        frame[y:y+h,x:x+w, c] = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + frame[y:y+h,x:x+w, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
    cv2.imwrite(r'D:\tmp\dataset\resultVideo\a' + str(num).zfill(5) + '.jpg', frame)
    num = num + 1
    pic = tkimage(frame)
    canvas.create_image(0, 0, anchor='nw', image=pic)
    myWindow.update()
    myWindow.after(1)
  video_captor.release()
  saveVideo(r'D:\tmp\dataset\resultVideo', num)
  print("合成完毕")
  # for x in list(locals().keys()):
  #   del locals()[x]
  # gc.collect()


# 清空画布，释放摄像头
def clear():
  global flag
  flag = True
  canvas.delete(tkinter.ALL)


def quit():
  ans = askyesno(title='Warning', message='Close the Window?')
  if ans:
    myWindow.destroy()
  else:
    return


# 在线视频单人替换
def main1():
  global flag
  flag = False
  demo(FLAGS.show_box)

def saveVideo(path, n):
  im_dir = path  # 图片存储路径
  video_dir = r'D:\tmp\dataset\re'+ str(n)+ '.avi'  # 合成后的视频名称, 只能合成avi格式视频
  imglist = sorted(os.listdir(im_dir))  # 将排序后的路径返回到imglist列表中
  img = cv2.imread(os.path.join(im_dir, imglist[0]))  # 合并目录与文件名生成图片文件的路径,随便选一张图片路径来获取图像大小
  H, W, D = img.shape  # 获取视频高\宽\深度
  print('height:' + str(H) + '--' + 'width:' + str(W) + '--' + 'depth:' + str(D))
  fps = 24  # 帧率一般选择20-30
  img_size = (W, H)  # 图片尺寸宽x高,必须是原图片的size,否则合成失败
  fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
  videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
  for image in imglist:
    img_name = os.path.join(im_dir, image)
    frame = cv2.imread(img_name)
    videoWriter.write(frame)
    print('合成==>' + img_name)
  videoWriter.release()
  print('finish!')
  del_files2(path)


def del_files2(dir_path):
  for root, dirs, files in os.walk(dir_path, topdown=False):
    # 第一步：删除文件
    for name in files:
      os.remove(os.path.join(root, name))  # 删除文件
    # 第二步：删除空文件夹
    for name in dirs:
      os.rmdir(os.path.join(root, name))  # 删除一个空目录


# 离线视频，多个人脸
def main2():
  global flag
  flag = False
  File = filedialog.askopenfilename(title='Choose a video.')
  video = cv2.VideoCapture(File)
  num = 0
  if video.isOpened():
    while flag==False:
      ret, img = video.read()  # img 就是一帧图片
      if not ret: break  # 当获取完最后一帧就结束
      faces = cascade_classifier.detectMultiScale(
        img,
        scaleFactor=1.2,
        minNeighbors=5
      )
      if not len(faces) > 0: continue
      max_are_face = faces[0]   # 找到最大的人脸
      for face in faces:
        [x, y, w, h] = face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 标记矩形框
        if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
          max_are_face = face
      for face in faces:
        if face[2]*face[3] == max_are_face[2]*max_are_face[3]: continue
        else:
          [xc, yc, wc, hc] = face
          faceImg = img[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
          faceImg = cv2.resize(faceImg, (48, 48), interpolation=cv2.INTER_CUBIC)
          tensor = image_to_tensor(faceImg)
          result = sess.run(probs, feed_dict={face_x: tensor})  # 代入训练网络
          if result is not None:
            emoji_face = feelings_faces[np.argmax(result[0])]
            for c in range(0, 3):
              emoji_face = cv2.resize(emoji_face, (wc, hc))
              img[yc:yc + hc, xc:xc + wc, c] = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + img[yc:yc + hc,
                                                xc:xc + wc, c] * (1.0 - emoji_face[:, :, 3] / 255.0)

      cv2.imwrite(r'D:\tmp\dataset\resultVideo\a' + str(num).zfill(5) + '.jpg', img)
      num = num + 1
      pic = tkimage(img)
      canvas.create_image(0, 0, anchor='nw', image=pic)
      myWindow.update()
      myWindow.after(1)
    saveVideo(r'D:\tmp\dataset\resultVideo', num)
    print("合成完毕")

  else:
    messagebox.showinfo('Reminder', 'Video opening failed.')


# 图片换脸，鼠标选择想换的人脸
def main3():
  # 选择本地图像
  File = filedialog.askopenfilename(title='Choose an image.')
  img = cv2.imread(File)
  print("原图大小：", img.shape[0])
  # 找人脸，返回vector保存各个人脸的坐标、大小
  faces = cascade_classifier.detectMultiScale(
    img,
    scaleFactor=1.2,
    minNeighbors=5
  )
  # 根据坐标，确定人脸图
  for face in faces:
    [x, y, w, h] = face
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)  # 标记矩形框

  pic = tkimage(img)
  canvas.image = pic
  canvas.create_image(0, 0, anchor='nw', image=pic)
  messagebox.showinfo('Reminder', 'Please select the face you want to replace.')

  # 根据鼠标选择的人像进行替换
  def func1(event):
    x1 = (img.shape[1]/550)*event.x    # 坐标转换
    y1 = (img.shape[0]/410)*event.y
    result = None
    for face in faces:
      [xc, yc, wc, hc] = face    # 人脸的坐标，以及大小
      if (x1>xc) & (x1<xc+wc ) & (y1>yc) & (y1<yc+hc):    # 判断是否在人脸区域
        faceImg = img[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
        faceImg = cv2.resize(faceImg, (48, 48), interpolation=cv2.INTER_CUBIC)
        tensor = image_to_tensor(faceImg)
        result = sess.run(probs, feed_dict={face_x: tensor})  # 代入训练网络，提取表情
        if result is not None:
          emoji_face = feelings_faces[np.argmax(result[0])]
          for c in range(0, 3):
            emoji_face = cv2.resize(emoji_face, (wc, hc))   # 用表情包替换人脸
            img[yc:yc + hc, xc:xc + wc, c] = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + img[yc:yc + hc, xc:xc + wc,c] * (
                                                   1.0 - emoji_face[:, :, 3] / 255.0)
    global flagsave    # 是否保存
    if flagsave:
      print("图像保存")
      path = r"D:/tmp/dataset/" +str(event.x) + ".jpg"  # 图片路径
      cv2.imwrite(path, img)  # 将图片保存
      flagsave = False
      print("图像已保存")
    pic = tkimage(img)    # 将替换后的图片显示在窗口
    canvas.image = pic
    canvas.create_image(0, 0, anchor='nw', image=pic)
    if result is None:   # 不在人脸区域，进行提醒
      messagebox.showinfo('Reminder', 'This is not a face.')
    # print(f"鼠标左键点击了一次坐标是:x={event.x} y={event.y}")

  canvas.bind("<Button-1>", func1)


def save():
  global flagsave
  flagsave = True


# 加载模型
tf.compat.v1.disable_eager_execution()
face_x = tf.placeholder(tf.float32, [None, 2304])
y_conv = deepnn(face_x)
probs = tf.nn.softmax(y_conv)
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
sess = tf.Session()
if ckpt and ckpt.model_checkpoint_path:
  saver.restore(sess, ckpt.model_checkpoint_path)
  print('Restore model sucsses!!')

# 表情图像
feelings_faces = []
for index, emotion in enumerate(EMOTIONS):
    feelings_faces.append(cv2.imread('./data/emojis/' + emotion + '.png', -1))

# 判断是否释放摄像
flag = False
flagsave = False
# 窗口
myWindow = Tk()
myWindow.title('Facial Information Protection System')
myWindow.resizable(width=True, height=True)
# myWindow.configure(bg='white')
width = 800
height = 600
screenwidth = myWindow.winfo_screenwidth()
screenheight = myWindow.winfo_screenheight()
alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
myWindow.geometry(alignstr)
canvas = Canvas(myWindow, bg='white', width=550, height=410)  # 绘制画布
canvas.place(x=220, y=70)
Label(myWindow, bg="lightblue", text='Face Replacement in Videos', font=("Arial", 10), width=22, height=1).place(x=20, y=50, anchor = 'nw')
Label(myWindow, bg="lightblue", text='Face Replacement in Images', font=("Arial", 10), width=22, height=1).place(x=20, y=210, anchor = 'nw')
# 按钮
put_pic_emo = tkinter.Button(myWindow, text='Real Time Video', font=('Arial', 9), width=15, height=1, command=main1)
put_pic_emo.place(relx=0.06, rely=0.16)
put_pic_emo = tkinter.Button(myWindow, text='Offline Video', font=('Arial', 9), width=15, height=1, command=main2)
put_pic_emo.place(relx=0.06, rely=0.23)
put_pic_emo = tkinter.Button(myWindow, text='Image', font=('Arial', 9), width=15, height=1, command=main3)
put_pic_emo.place(relx=0.06, rely=0.42)
put_pic_emo = tkinter.Button(myWindow, text='Save', font=('Arial', 9), width=15, height=1, command=save)
put_pic_emo.place(relx=0.06, rely=0.49)
put_pic_emo = tkinter.Button(myWindow, text='Clear', font=('Arial', 9), width=10, height=1, command=clear)
put_pic_emo.place(relx=0.03, rely=0.73)
put_pic_emo = tkinter.Button(myWindow, text='Quit', font=('Arial', 9), width=10, height=1, command=quit)
put_pic_emo.place(relx=0.15, rely=0.73)

myWindow.mainloop()