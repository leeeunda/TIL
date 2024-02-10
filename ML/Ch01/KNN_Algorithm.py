{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 도미 데이터\n",
    "bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, \n",
    "                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, \n",
    "                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]\n",
    "bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, \n",
    "                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, \n",
    "                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "plt.scatter(bream_length, bream_weight)\n",
    "plt.xlabel('length') # x축은 길이\n",
    "plt.ylabel('weight') # y축은 무게\n",
    "\n",
    "# 빙어 데이터\n",
    "smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]\n",
    "smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]\n",
    "\n",
    "# 산점도 그래프로 나타내기\n",
    "plt.scatter (bream_length, bream_weight)\n",
    "plt.scatter(smelt_length, smelt_weight)\n",
    "plt.xlabel('length')\n",
    "plt.ylabel('weight')\n",
    "plt.show()\n",
    "\n",
    "# 데이터 합치기 : 2차원 리스트 생성\n",
    "length=bream_length+smelt_length\n",
    "weight=bream_weight+smelt_weight\n",
    "\n",
    "fish_data=[[l,w] for l,w in zip(length,weight)]\n",
    "print(fish_data)\n",
    "\n",
    "# 정답 데이터\n",
    "fish_target=[1]*35 + [0]*14\n",
    "print(fish_target)\n",
    "\n",
    "from sklearn.neighbors import KNeighborsClassifier # KNN 알고리즘 구현 클래스\n",
    "kn=KNeighborsClassifier() \n",
    "kn.fit(fish_data, fish_target) # 훈련\n",
    "kn.score(fish_data, fish_target) # 평가\n",
    "kn.predict([[30,600]]) # 예측\n",
    "\n",
    "\n",
    "kn49=KNeighborsClassifier(n_neighbors=49)\n",
    "kn49.fit(fish_data,fish_target)\n",
    "kn49.score(fish_data,fish_target)"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
