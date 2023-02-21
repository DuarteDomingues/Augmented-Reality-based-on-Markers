import cv2 
import numpy as np
from camera_calibration import calibrate_camera,do_undistortion

#funcao cria um marcador aruco com parametros especificos
def create_marker(dic_type,id,img_size,border_bits, marker_image_name):

    #carregar o dicionario utilizado para carregar os markers
    dic = cv2.aruco.Dictionary_get(dic_type)
    #inicliazar os detector parameters
    marker = np.zeros((img_size, img_size), dtype=np.uint8)
    marker = cv2.aruco.drawMarker(dic, id, img_size, marker, border_bits)
    cv2.imwrite(marker_image_name, marker)


#funcao para detetar marcadores em imagens
def detect_markers(dic_type, img, draw_box=False):
    #trasnformar imagem para grayscale 
    #img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #obter o dicionario aruco
    dic = cv2.aruco.Dictionary_get(dic_type)
    #inicializar  os parametros do detetor
    params = cv2.aruco.DetectorParameters_create()
    #detetar os marcadores na imagem
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, dic, parameters = params)
    print(ids)
    #desenhar bounding box no marcador
    if draw_box:
        img = cv2.aruco.drawDetectedMarkers(img, corners,ids) 
    
        #cv2.imwrite('cona.png', img)
    
    return corners,ids,img



'''

dic_type = cv2.aruco.DICT_4X4_50
create_marker(dic_type,10,200,1, "marker_6.png")

marker_size=4
total_markers=50
img ="teste.png"

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints,imgpoints,gray_img = calibrate_camera(7,7,criteria,'./imgs/calibration_3/*.jpg')
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_img.shape[::-1], None, None)


cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    img = do_undistortion(img, mtx,dist)
    c,d =  detect_markers(dic_type,img,draw_box=True)
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

'''