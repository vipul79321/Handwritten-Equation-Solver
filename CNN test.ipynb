{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "E:\\apps\\anaconda2\\lib\\site-packages\\h5py\\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.\n",
      "  from ._conv import register_converters as _register_converters\n",
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "import cv2\n",
    "import numpy\n",
    "from keras.datasets import mnist\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense\n",
    "from keras.layers import Dropout\n",
    "from keras.layers import Flatten\n",
    "from keras.layers.convolutional import Conv2D\n",
    "from keras.layers.convolutional import MaxPooling2D\n",
    "from keras.utils import np_utils\n",
    "from keras import backend as K\n",
    "K.set_image_dim_ordering('th')\n",
    "from keras.models import model_from_json\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "json_file = open('model_final.json', 'r')\n",
    "loaded_model_json = json_file.read()\n",
    "json_file.close()\n",
    "loaded_model = model_from_json(loaded_model_json)\n",
    "# load weights into new model\n",
    "loaded_model.load_weights(\"model_final.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "img = cv2.imread('C:/Users/Vipul Gupta/Pictures/test.jpeg',cv2.IMREAD_GRAYSCALE)\n",
    "#kernel = np.ones((3,3),np.uint8)\n",
    "cv2.imshow(\"wo\",img)\n",
    "cv2.waitKey(0)\n",
    "cv2.destroyAllWindows()\n",
    "#erosion = cv2.erode(img,kernel,iterations = 3)\n",
    "#dilation = cv2.dilate(img,kernel,iterations = 1)\n",
    "#img=dilation\n",
    "if img is not None:\n",
    "    #images.append(img)\n",
    "    img=~img\n",
    "    ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)\n",
    "    ret,ctrs,ret=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)\n",
    "    cnt=sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])\n",
    "    w=int(28)\n",
    "    h=int(28)\n",
    "    train_data=[]\n",
    "    #print(len(cnt))\n",
    "    rects=[]\n",
    "    for c in cnt :\n",
    "        x,y,w,h= cv2.boundingRect(c)\n",
    "        rect=[x,y,w,h]\n",
    "        rects.append(rect)\n",
    "    #print(rects)\n",
    "    bool_rect=[]\n",
    "    for r in rects:\n",
    "        l=[]\n",
    "        for rec in rects:\n",
    "            flag=0\n",
    "            if rec!=r:\n",
    "                if r[0]<(rec[0]+rec[2]+10) and rec[0]<(r[0]+r[2]+10) and r[1]<(rec[1]+rec[3]+10) and rec[1]<(r[1]+r[3]+10):\n",
    "                    flag=1\n",
    "                l.append(flag)\n",
    "            if rec==r:\n",
    "                l.append(0)\n",
    "        bool_rect.append(l)\n",
    "    #print(bool_rect)\n",
    "    dump_rect=[]\n",
    "    for i in range(0,len(cnt)):\n",
    "        for j in range(0,len(cnt)):\n",
    "            if bool_rect[i][j]==1:\n",
    "                area1=rects[i][2]*rects[i][3]\n",
    "                area2=rects[j][2]*rects[j][3]\n",
    "                if(area1==min(area1,area2)):\n",
    "                    dump_rect.append(rects[i])\n",
    "    #print(len(dump_rect)) \n",
    "    final_rect=[i for i in rects if i not in dump_rect]\n",
    "    #print(final_rect)\n",
    "    for r in final_rect:\n",
    "        x=r[0]\n",
    "        y=r[1]\n",
    "        w=r[2]\n",
    "        h=r[3]\n",
    "        im_crop =thresh[y:y+h+10,x:x+w+10]\n",
    "        \n",
    "\n",
    "        im_resize = cv2.resize(im_crop,(28,28))\n",
    "        cv2.imshow(\"work\",im_resize)\n",
    "        cv2.waitKey(0)\n",
    "        cv2.destroyAllWindows()\n",
    "\n",
    "        im_resize=np.reshape(im_resize,(1,28,28))\n",
    "        train_data.append(im_resize)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "7+3\n"
     ]
    }
   ],
   "source": [
    "s=''\n",
    "for i in range(len(train_data)):\n",
    "    train_data[i]=np.array(train_data[i])\n",
    "    train_data[i]=train_data[i].reshape(1,1,28,28)\n",
    "    result=loaded_model.predict_classes(train_data[i])\n",
    "    if(result[0]==10):\n",
    "        s=s+'-'\n",
    "    if(result[0]==11):\n",
    "        s=s+'+'\n",
    "    if(result[0]==12):\n",
    "        s=s+'*'\n",
    "    if(result[0]==0):\n",
    "        s=s+'0'\n",
    "    if(result[0]==1):\n",
    "        s=s+'1'\n",
    "    if(result[0]==2):\n",
    "        s=s+'2'\n",
    "    if(result[0]==3):\n",
    "        s=s+'3'\n",
    "    if(result[0]==4):\n",
    "        s=s+'4'\n",
    "    if(result[0]==5):\n",
    "        s=s+'5'\n",
    "    if(result[0]==6):\n",
    "        s=s+'6'\n",
    "    if(result[0]==7):\n",
    "        s=s+'7'\n",
    "    if(result[0]==8):\n",
    "        s=s+'8'\n",
    "    if(result[0]==9):\n",
    "        s=s+'9'\n",
    "    \n",
    "print(s)    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "10"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "eval(s)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
