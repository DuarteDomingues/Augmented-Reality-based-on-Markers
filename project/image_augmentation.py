import cv2
from matplotlib.pyplot import draw 
import numpy as np
from camera_calibration import calibrate_camera,do_undistortion
from detection_pose_camera import detect_markers

def warp_image(corners, id,img,imgAug,draw=False):

    #obter posicao dos markers
    top_left = corners[0][0][0], corners[0][0][1]
    top_right = corners[0][1][0], corners[0][1][1]
    bot_right = corners[0][2][0], corners[0][2][1]
    bot_left = corners[0][3][0], corners[0][3][1]

    #obter h,w,chanel da imagem para ser augmentada
    h, w, c = imgAug.shape

    pts1 = np.array([top_left, top_right, bot_right, bot_left])
    pts2 = np.float32([[0,0], [w,0], [w,h], [0,h]])


    mtx, _ = cv2.findHomography(pts2, pts1)
    imgout = cv2.warpPerspective(imgAug, mtx, (img.shape[1], img.shape[0]))

    cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))
    #imgout = img + imgout
    imgout = img 
    return imgout


def render_image(pts, img, imgAug):

    #obter h,w,chanel da imagem para ser augmentada
    h, w, c = imgAug.shape

    pts2 = np.float32([[0,0], [w,0], [w,h], [0,h]])

    mtx, _ = cv2.findHomography(pts2, pts)
    imgout = cv2.warpPerspective(imgAug, mtx, (img.shape[1], img.shape[0]))

    cv2.fillConvexPoly(img, pts.astype(int), (0, 0, 0))
    imgout = img + imgout
    return imgout


def render_cubes(corners, mtx, dist, img,imgAug,drawLine = False):

    #retorna vetores de rotacao e translacao do marcador
    #transforms pontos de cada sistema de coordenadas do marcador, para coords da camara
    objPoints =  np.float32([[-0.1,-0.1,0.2],[-0.1,-0.1,0],[0.1,-0.1,0],[0.1,-0.1,0.2], 
    [-0.1,0.1,0.2], [-0.1,0.1,0],[0.1,0.1,0],[0.1,0.1,0.2],
    [0.1,-0.1,0], [0.1,0.1,0], [-0.1,0.1,0],[-0.1,-0.1,0],
    [0.1,-0.1,0.2], [0.1,0.1,0.2],[-0.1,0.1,0.2],[-0.1,-0.1,0.2],
    [0.1,-0.1,0.2],[0.1,-0.1,0],[0.1,0.1,0],[0.1,0.1,0.2],
    [-0.1,-0.1,0.2],[-0.1,-0.1,0],[-0.1,0.1,0],[-0.1,0.1,0.2]
    ]).reshape(-1,3)

    #projetar pontos 3D para plano de imagem
    rvec, tvec,_ = cv2.aruco.estimatePoseSingleMarkers(corners,0.2,mtx,dist)
    imgPoints,jac = cv2.projectPoints(objPoints,rvec, tvec, mtx, dist)

    square_face_1= np.asarray([imgPoints[i][0].ravel().astype(int) for i in range(4)])
    square_face_2= np.asarray([imgPoints[i][0].ravel().astype(int) for i in range(4,8)])
    square_face_3= np.asarray([imgPoints[i][0].ravel().astype(int) for i in range(8,12)])
    square_face_4= np.asarray([imgPoints[i][0].ravel().astype(int) for i in range(12,16)])
    square_face_5= np.asarray([imgPoints[i][0].ravel().astype(int) for i in range(16,20)])
    square_face_6= np.asarray([imgPoints[i][0].ravel().astype(int) for i in range(20,24)])

    #DESENHAR IMAGENS NAS FACES
    if (drawLine==False):
    
        img = render_image(square_face_3,img,imgAug)  
        img = render_image(square_face_4,img,imgAug) 
        img = render_image(square_face_1,img,imgAug)  
        img = render_image(square_face_2,img,imgAug)  
        img = render_image(square_face_5,img,imgAug)  
        img = render_image(square_face_6,img,imgAug)  
    
    #DESENHAR LINHAS
    else:
        colors_list= [(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255)]
        c=0
        for i in range (0,24,4):
            
            cv2.line(img, imgPoints[i][0].ravel().astype(int), imgPoints[i+3][0].ravel().astype(int), color=colors_list[c])
            cv2.line(img, imgPoints[i][0].ravel().astype(int), imgPoints[i+1][0].ravel().astype(int), color=colors_list[c])
            cv2.line(img, imgPoints[i+1][0].ravel().astype(int), imgPoints[i+2][0].ravel().astype(int), color=colors_list[c])
            cv2.line(img, imgPoints[i+2][0].ravel().astype(int), imgPoints[i+3][0].ravel().astype(int), color=colors_list[c])
            c=c+1


    return img



def render_piramids(corners, mtx, dist, img):

    #retorna vetores de rotacao e translacao do marcador
    #transforms pontos de cada sistema de coordenadas do marcador, para coords da camara
    objPoints =  np.float32([
    [0.1,-0.1,0], [0.1,0.1,0], [-0.1,0.1,0],[-0.1,-0.1,0],
    [0.1,-0.1,0],[0,0,0.1],[-0.1,-0.1,0],
    [0.1,0.1,0],[0,0,0.1],[-0.1,0.1,0],
    [0.1,0.1,0],[0,0,0.1],[0.1,0.1,0],
    [-0.1,-0.1,0],[0,0,0.1],[-0.1,0.1,0],


    ]).reshape(-1,3)

    #projetar pontos 3D para plano de imagem
    rvec, tvec,_ = cv2.aruco.estimatePoseSingleMarkers(corners,0.2,mtx,dist)
    imgPoints,jac = cv2.projectPoints(objPoints,rvec, tvec, mtx, dist)

    #BASE
    cv2.line(img, imgPoints[0][0].ravel().astype(int), imgPoints[3][0].ravel().astype(int), color=(255,0,0) )
    cv2.line(img, imgPoints[0][0].ravel().astype(int), imgPoints[1][0].ravel().astype(int), color=(255,0,0) )
    cv2.line(img, imgPoints[1][0].ravel().astype(int), imgPoints[2][0].ravel().astype(int), color=(255,0,0) )
    cv2.line(img, imgPoints[2][0].ravel().astype(int), imgPoints[3][0].ravel().astype(int), color=(255,0,0))

    cv2.line(img, imgPoints[4][0].ravel().astype(int), imgPoints[5][0].ravel().astype(int), color=(0,255,0) )
    cv2.line(img, imgPoints[5][0].ravel().astype(int), imgPoints[6][0].ravel().astype(int), color=(0,255,0) )

    cv2.line(img, imgPoints[7][0].ravel().astype(int), imgPoints[8][0].ravel().astype(int), color=(0,255,0) )
    cv2.line(img, imgPoints[8][0].ravel().astype(int), imgPoints[9][0].ravel().astype(int), color=(0,255,0) )

    cv2.line(img, imgPoints[10][0].ravel().astype(int), imgPoints[11][0].ravel().astype(int), color=(0,255,0) )
    cv2.line(img, imgPoints[11][0].ravel().astype(int), imgPoints[12][0].ravel().astype(int), color=(0,255,0) )

    cv2.line(img, imgPoints[13][0].ravel().astype(int), imgPoints[14][0].ravel().astype(int), color=(0,0,255) )
    cv2.line(img, imgPoints[14][0].ravel().astype(int), imgPoints[15][0].ravel().astype(int), color=(0,0,255) )

    return img



#definir tipo de dicionario do marcador
dic_type = cv2.aruco.DICT_4X4_50

#termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints,imgpoints,gray_img = calibrate_camera(7,7,criteria,'./imgs/calibration_3/*.jpg')
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_img.shape[::-1], None, None)

#iniciar captura de video
cap = cv2.VideoCapture(0)

#ler imagens
imgAug_cat = cv2.imread('./imgs/cat.jpg')
imgAug_cube_1 = cv2.imread('./imgs/cube2.jpg')
imgAug_cube_2 = cv2.imread('./imgs/cube5.jpg')
imgAug_cube_3 = cv2.imread('./imgs/cube4.jpg')

while True:
    success, img = cap.read()
    #img_undist = img
    img = do_undistortion(img, mtx,dist)
    markers,ids, img = detect_markers(dic_type, img, draw_box=True)

    # loop through all the markers and augment each one
    if len(markers)!=0:
        for bbox, id in zip(markers,ids):
            #img = warp_image(bbox, id, img, imgAug)
            
            #consoante o id do marcador, dar load de um cubo diferente
            if (id==5):
                #img = render_cubes(bbox, mtx, dist, img,imgAug_cube_1)
                img = render_cubes(bbox, mtx, dist, img,imgAug_cube_1)
            elif (id==6):
                #img = render_cubes(bbox, mtx, dist, img,imgAug_cube_2)
                img = render_cubes(bbox, mtx, dist, img,imgAug_cube_2,drawLine=True)
            elif (id==7):

                #img = render_cubes(bbox, mtx, dist, img,imgAug_cube_3)
                img = render_piramids(bbox, mtx, dist, img)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
