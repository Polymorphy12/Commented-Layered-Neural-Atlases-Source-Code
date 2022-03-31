import matplotlib as mpl

mpl.use('Agg')
import skimage.metrics
import io
from scipy.interpolate import griddata
import matplotlib.image as mpimg
import torch
import colorsys
from pathlib import Path
from loss_utils import get_rigidity_loss, get_optical_flow_loss_all, get_optical_flow_alpha_loss_all

import matplotlib.pyplot as plt
import skimage.measure
import os
from PIL import Image
import cv2
import numpy as np
import imageio


'''
# taken from https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e
# sample coordinates x,y from image im.

interpolation은 알려진 값들 사이(중간)에 위치한 값을 알려진 값으로부터 추정하는 것을 말한다.
(extrapolation은 알려진 값으로부터 범위 밖 외부 값을 추정하는 것을 말한다.)
아래 코드는 이미지 im으로부터 x,y좌표를 샘플링한다.

인풋으로 들어오는 x와 y는 벡터들(텐서들?)이다.
np.floor(x) 함수는 x의 모든 원소들을 내림한 벡터를 반환한다.

numpy.clip(array, min, max) 함수는
array 내의 element들에 대해서
min 값보다 작은 값들을 min 값으로 바꿔주고
max값 보다 큰 값들을 max값으로 바꿔준다.
'''
def bilinear_interpolate_numpy(im, x, y):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    # 좌표 벡터들이 이미지 픽셀 범위값을 벗어나지 않도록 한다.
    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    # 이미지에 좌표의 벡터를 넣으면
    # 아웃풋은 결과값 (좌표에 해당하는 픽셀값) 벡터가 된다.
    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return (Ia.T * wa).T + (Ib.T * wb).T + (Ic.T * wc).T + (Id.T * wd).T


'''
Given target (non integer) uv coordinates with corresponding alpha values,
create an 1000 * 1000 uv image of alpha values

np.linspace(start, end, num)는 
start 숫자로 시작해서 end 숫자로 끝나도록, 
일정한 간격으로 num 개수 만큼 띄워진 숫자들 벡터를 반환한다.
ex) np.linspace(0, 999, 1000)
-> array([0, 1, 2, ... , 999])
ex) np.linspace(2.0, 3.0, 5)
-> array([2. , 2.25, 2.5, 2.75, 3.])

xv, yv = np.meshgrid(x, y, indexing="xy")가 있고
iv, jv = np.meshgrid(x, y, indexing="ij")가 있다.
이 때
meshgrid는 좌표 벡터들로부터 좌표 행렬을 만들어내는데,
xv와 iv 는 transpose 관계다.
yv와 jv는 transpose 관계다.

scipy.interpolate의 griddata는
데이터 좌표, 데이터 값, 보간법을 적용할 점들을 인풋으로 받아서
보간법(interpolation)을 적용한 값들을 행렬형태로 반환한다.
'''
def interpolate_alpha(m1_alpha_v, m1_alpha_u, m1_alpha_alpha):
    xv, yv = np.meshgrid(np.linspace(0, 999, 1000), np.linspace(0, 999, 1000))

    m1_alpha_u = np.concatenate(m1_alpha_u)
    m1_alpha_v = np.concatenate(m1_alpha_v)
    m1_alpha_alpha = np.concatenate(m1_alpha_alpha)

    masks1 = griddata((m1_alpha_u, m1_alpha_v), m1_alpha_alpha,
                      (xv, yv), method='linear')
    return masks1

'''
Given uv points in the range (-1,1) and an image (with a given "resolution") 
that represents a crop (defined by "minx", "maxx", "miny", "maxy")

Change uv points to pixel coordinates, and sample points from the image

<인풋에 대하여>
resolution : 아래에 있는 함수 사용례를 보면 정수를 쓴다.
             그런데 이미지 해상도를 표현할 때는 가로와 높이를 명시하지 않나?
             어떻게 하다가 단일 정수가 되는건지 궁금하다.
             -> 아래를 보면 1000*1000 이미지임을 알 수 있음.
             mx_logicalx, mx_logicaly 조건문을 볼 것.

minx, maxx, miny, maxy : 이미지를 crop할 때 사용하는 값.

pointx, pointy : u,v 좌표 

image : 복원한 atlas texture를 사용한다.

<여기서 사용한 외부 함수에 대해>
np.logical_and(x1, x2) : 조건문 x1, x2가 모두 참인 원소는 True,
                         아닌경우 False로 채워진 텐서를 반환한다.
                         여기서 조건문 x1, x2도 각각 텐서에 대한 조건문이다.
                         즉, 각각 True, False를 원소로 채운 텐서가 된다.
                         출력값은 x1, x2와 같은 크기를 가진 텐서가 된다.
'''
def get_colors(resolution, minx, maxx, miny, maxy, pointx, pointy, image):
    pixel_size = resolution / (maxx - minx)
    # Change uv to pixel coordinates of the discretized image
    pointx2 = ((pointx - minx) * pixel_size).numpy()
    pointy2 = ((pointy - miny) * pixel_size).numpy()

    # Bilinear interpolate pixel colors from the image
    # 형태가 어떻게 생겼는지 보고싶다.
    pixels = bilinear_interpolate_numpy((image.numpy(), pointx2, pointy2))

    # 아래 세 문단은 pixel의 각 값들이 주석이 요구하는 조건에 부합하는지 True, False로 마킹한다.
    # 위치가 양수일것, 그리고 이미지 범위 내에 포함되어 있을 것.

    # Relevant pixel locations should be positive:
    pos_logicaly = np.logical_and(np.ceil(pointy2) >= 0, np.floor(pointy2) >= 0)
    pos_logicalx = np.logical_and(np.ceil(pointx2) >= 0, np.floor(pointx2) >= 0)
    pos_logical = np.logical_and(pos_logicaly, pos_logicalx)

    # Relevant pixel locations should be inside the image borders:
    mx_logicaly = np.logical_and(np.ceil(pointy2) < resolution, np.floor(pointy2) < resolution)
    mx_logicalx = np.logical_and(np.ceil(pointx2) < resolution, np.floor(pointx2) < resolution)
    mx_logical = np.logical_and(mx_logicaly, mx_logicalx)

    # Relevant should satisfy both conditions
    relevant = np.logical_and(pos_logical, mx_logical)

    # Relevant는 모두 조건을 만족하는지 판단하는 True, False로 채워져 있다.
    # 이 때 pixels[relevant]와 같이 사용하면
    # True에 해당하는 pixels의 원소들은 output에 포함하고,
    # False에 해당하는 pixels의 원소들은 output에 포함하지 않는다.
    # Q. 이 때 픽셀의 각 원소는 RGB값인가?
    # A. 맞는 듯 아래 사용례를 볼 것.
    return pixels[relevant], pointx2[relevant], pointy2[relevant], relevant


'''
Sample discrete atlas image from a neural atlas
'''
def get_high_res_texture(resolution, minx, maxx, miny, maxy, model_F_atlas, device):
    # torch.linspace는 np.linspace와 같은 역할을 한다.
    indsx = torch.linspace(minx, maxx, resolution)
    indsy = torch.linspace(miny, maxy, resolution)
    # 복원한 이미지의 W,H 값이 모두 resolution값을 가짐을 알 수 있다.
    reconstruction_texture2 = torch.zeros((resolution, resolution, 3))
    counter = 0
    with torch.no_grad():
        # reconstruct image row by row
        for i in indsy:
            # counter 변수는 이미지 row 번호를 의미한다.
            # RGB 값을 예측하는 model_F_atlas에
            # resolution 크기 만큼의 u,v 좌표값 벡터를 만들어 넘겨주기 때문에
            # model_F_atlas가
            # resolution 크기 만큼의 RGB 벡터를 출력해서 텍스쳐에 저장한다.
            reconstruction_texture2[counter, :, :] = model_F_atlas(
                torch.cat((indsx.unsqueeze(1), i * torch.ones_like(indsx.unsqueeze(1))),
                          dim=1).to(device)).detach().cpu()
            counter = counter + 1
        # move colors to RGB color domain (0,1)
        reconstruction_texture2 = 0.5 * (reconstruction_texture2 + 1)

        reconstruction_texture2_orig = reconstruction_texture2.clone()

        # Add text pattern to the texture, in order to visualize the mapping functions.
        for ii in range(40, 500, 80):
            cur_color = colorsys.hsv_to_rgb((ii - 40) / 500, 1.0, 1.0)
            cv2.putText(reconstruction_texture2.numpy(),
                        'abcdefghijlmnopqrstuvwxyz1234567890!@#$%^&*()-+=>', (10, ii),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, cur_color, 2, cv2.LINE_AA)
            cv2.putText(reconstruction_texture2.numpy(),
                        'ABCDEFGHIJKLMNOPQRSTUVWXYZ?~;:<./\|][{},', (10, ii + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, cur_color, 2, cv2.LINE_AA)

        for ii in range(40, 500, 80):
            cur_color = colorsys.hsv_to_rgb((ii - 40) / 500, 1.0, 1.0)
            cv2.putText(reconstruction_texture2.numpy(),
                        'abcdefghijlmnopqrstuvwxyz1234567890!@#$%^&*()-+=>', (10, ii + 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, cur_color, 2, cv2.LINE_AA)
            cv2.putText(reconstruction_texture2.numpy(),
                        'ABCDEFGHIJKLMNOPQRSTUVWXYZ?~;:<./\|][{},',
                        (10, ii + 40 + 500), cv2.FONT_HERSHEY_SIMPLEX, 1.1, cur_color, 2,
                        cv2.LINE_AA)

        return reconstruction_texture2, reconstruction_texture2_orig

'''
pyplot으로 그려놓은 그림을 
데이터 배열 이미지 형태로 변환해서 반환한다.

<입력값에 대해>
fig : matplotlib.pyplot의 figure
dpi : 이미지의 해상도
'''
def get_img_from_fig(fig, dpi=180):
    # 메모리에서 Bytes 데이터를 조작하기 위한 함수
    buf = io.BytesIO()
    # 이렇게 하면 인메모리 buf에 이미지를 저장한다.
    fig.savefig(buf, format="png", dpi = dpi)
    # stream 위치를 시작위치로 설정한다.
    buf.seek(0)
    # np.frombuffer()는
    # 바이너리(버퍼) 데이터를 1차원 배열로 만들어주는 역할을 한다.
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

'''
Given a mapping model "model_F_mapping", 
maskRCNN masks "mask_frames" and alpha network "F_alpha", 
find a range in the uv domain, 
that maskRCNN points are mapped to by model_F_mapping.

model_F_mapping 모델을 사용해서
maskRCNN 픽셀들을 uv 도메인으로 매핑한다.
이 때 uv 도메인의 범위를 구한다.
'''
def get_mapping_area(model_F_mapping, F_alpha, mask_frames, resx, number_of_frames,
                     uv_shift, device, invert_alpha=False, alpha_thresh=-0.5):
    # consider only pixels that their masks are 1
    relis_i, relis_j, relis_f = torch.where(mask_frames)

    # 위 코드를 설명하는 주석.
    # torch.where(조건문 텐서)는
    # torch.nonzero(조건문 텐서, as_tuple=True) 와 같은 결과를 낸다.
    # 이 때 조건문 텐서에 일반적인 텐서를 넣으면
    # 원소가 0이 아닌 부분의 좌표(indices)를 출력한다.
    # 예를들어 아래의 출력 결과는 다음과 같다.
    # torch.nonzero(
    #   torch.tensor(
    #       [[1,1,0,0],
    #        [0,1,0,0],
    #        [0,0,1,0],
    #        [0,0,0,1]]), as_tuple=True)
    # 출력 결과:
    #   (tensor([0, 0, 1, 2, 3]),
    #    tensor([0, 1, 1, 2, 3]))
    # 여기서 첫번째 텐서는 row (i),
    # 두 번째 텐서는 column (j)를 나타낸다.

    # split all i, j, f coordinates to batches of size 100k
    relisa = np.array_split(relis_i.numpy(), np.ceil(relis_i.shape[0] / 100000))
    reljsa = np.array_split(relis_j.numpy(), np.ceil(relis_i.shape[0] / 100000))
    relfsa = np.array_split(relis_f.numpy(), np.ceil(relis_i.shape[0] / 100000))

    minx = 1
    miny = 1
    maxx = -1
    maxy = -1
    with torch.no_grad():
        for i in range(len(relisa)):
            # i, j, f가 [-1, 1] 사이의 값을 가지도록 normalize 해준다.
            # 즉, 논문에서의 x,y,t가 되도록 한다.
            relis = torch.from_numpy(relisa[i]).unsqueeze(1) / (resx / 2) - 1
            reljs = torch.from_numpy(reljsa[i]).unsqueeze(1) / (resx / 2) - 1
            relfs = torch.from_numpy(relfsa[i]).unsqueeze(1) / (number_of_frames / 2) - 1

            # 모델에 x,y,t를 통과시켜 uv값, alpha값을 얻는다.
            uv = model_F_mapping(torch.cat((reljs, relis, relfs),
                                           dim=1).to(device)).cpu()
            alpha = F_alpha(torch.cat((reljs, relis, relfs),
                                      dim=1).to(device)).cpu().squeeze()

            if invert_alpha:
                alpha = -alpha
            if torch.any(alpha > alpha_thresh):
                uv = uv * 0.5 + uv_shift
                # "alpha > alpha_thresh" 조건문은 True,False가 원소인 텐서가 된다.
                # "uv[alpha > alpha_thresh, 0]"는
                # 조건문에 해당하는 벡터의 0번째 원소들로 만든 벡터이다.
                # 여기에 torch.min을 적용하면 벡터 내의 최소값인 원소를 내놓는다.
                # torch.max를 적용하면 벡터 내의 최대값인 원소를 내놓는다.
                curminx = torch.min(uv[alpha > alpha_thresh, 0])
                curminy = torch.min(uv[alpha > alpha_thresh, 1])
                curmaxx = torch.max(uv[alpha > alpha_thresh, 0])
                curmaxy = torch.max(uv[alpha > alpha_thresh, 1])

                minx = torch.min(torch.tensor([curminx, minx]))
                miny = torch.min(torch.tensor([curminy, miny]))
                maxx = torch.max(torch.tensor([curmaxx, maxx]))
                maxy = torch.max(torch.tensor([curmaxy, maxy]))

    maxx = np.minimum(maxx, 1)
    maxy = np.minimum(maxy, 1)

    minx = np.maximum(minx, -1)
    miny = np.maximum(miny, -1)

    # Q. 정사각형이 아닌건가?
    edge_size = torch.max(torch.tensor([maxx - minx, maxy - miny]))

    return maxx, minx, maxy, miny, edge_size


'''
for visualizing uv images 
their values are mapped to the range (0,1) 
by using (edge_size, minx, miny)
which represent the information about the mapping range. 

The idea is to stretch this range to (0,1).

<입력값에 대해>
uv_frames_reconstruction : (resy, resx, 3, number_of_frames)
                            여기서 "3"이 뭐지?
                            저번에 jif_foreground처럼 uv좌표 + frame_num인가.
                            그렇다면 다른 차원들 resy, resx, number_of_frames랑 차이가 뭐지?
                            -> 
                            resy, resx, number_of_frames는 말 그대로 그 갯수만큼 원소가 있다는 이야기고.
                            3은 실질적인 내용 (좌표 + framenum)을 담는다고 이해하면 될까.
                            
                            RGB값일까? 좌표 + frame_num일까?
                            아래에 사용례를 보면 0,1,2 중에 0,1은 좌표일거라는 생각이 드는데.
values_shift : 
edge_size :
minx :
miny :
'''
def normalize_uv_images(uv_frames_reconstruction, values_shift, edge_size, minx, miny):
    # uv 좌표를 normalize한다.
    uv_frames_reconstruction[:, :, 0, :] = ((uv_frames_reconstruction[:, :, 0, :] * 0.5 + values_shift) - np.float64(
        minx)) /edge_size
    uv_frames_reconstruction[:, :, 1, :] = ((uv_frames_reconstruction[:, :, 1, :] * 0.5 + values_shift) - np.float64(
        miny)) / edge_size
    # 원소값을 범위[0,1] 내로 맞춰준다.
    uv_frames_reconstruction[uv_frames_reconstruction > 1] = 1
    uv_frames_reconstruction[uv_frames_reconstruction < 0] = 0
    return uv_frames_reconstruction


'''
모델을 평가할 때는 특정 metric을 정의해서 기준 삼는다.
예를들어 분류 모델은 Accuracy, Precision, Recall을 기준으로 평가한다.

이번 모델에서는
loss_util 파일에서 정의한 loss들을 metric으로 삼는다.
(RGB error, Rigidity loss, flow_alpha_loss, flow_loss)
그리고 alpha값과 mask rcnn에서 추출한 마스크를 비교한다.
비디오를 어떻게 복원하는지도 확인한다.

이런 기준들을 matplotlib로 그림을 그려 확인한다.
'''
def evaluate_model(model_F_atlas, resx, resy, number_of_frames, model_F_mapping1,
                   model_F_mapping2, model_alpha, video_frames, results_folder,
                   iteration, mask_frames, optimizer_all, writer, vid_name,
                   derivative_amount, uv_mapping_scale, optical_flows, optical_flows_mask, device,
                   save_checkpoint=True, show_atlas_alpha=False):

    os.mkdir(os.path.join(results_folder, '%06d' % iteration))
    evaluation_folder = os.path.join(results_folder, '%06d' % iteration)
    resx = np.int64(resx)
    resy = np.int64(resy)
    larger_dim = np.maximum(resx, resy)
    if save_checkpoint:
        # state_dict()는 모델의 파라미터 텐서를 매핑하는 파이썬 dictionary이다.
        torch.save({
            'F_atlas_state_dict': model_F_atlas.state_dict(),
            'iteration': iteration,
            'model_F_mapping1_state_dict': model_F_mapping1.state_dict(),
            'model_F_mapping2_state_dict': model_F_mapping2.state_dict(),
            'model_F_alpha_state_dict': model_alpha.state_dict(),
            'optimizer_all_state_dict': optimizer_all.state_dict()
        }, '%s/checkpoint' % (results_folder))

    # get relevant bounding box in the foreground atlas coordinates, using the input masks.
    minx = 0
    miny = 0
    edge_size = 1
    maxx2, minx2, maxy2, miny2, edge_size2 = get_mapping_area(model_F_mapping2,
                                                             model_alpha, mask_frames > -1,
                                                             larger_dim,
                                                             number_of_frames,
                                                             torch.tensor([-0.5, -0.5]), device, invert_alpha=True)
    maxxt, minxt, maxyt, minyt, edge_sizet = get_mapping_area(model_F_mapping1, model_alpha, mask_frames > 0.5,
                                                              larger_dim,
                                                              number_of_frames,
                                                              torch.tensor([0.5, 0.5]), device, invert_alpha=False,
                                                              alpha_thresh=0.95)

    edited_tex1, texture_orig1 = get_high_res_texture(
        1000,
        minx, minx + edge_size, miny, miny + edge_size, model_F_atlas, device
    )

    edited_tex2, texture_orig2 = get_high_res_texture(
        1000,
        minx2, minx2 + edge_size2, miny2, miny2 + edge_size2, model_F_atlas, device
    )

    _, texture_orig1t = get_high_res_texture(
        500,
        minxt, minxt + edge_sizet, minyt, minyt + edge_sizet, model_F_atlas, device
    )

    _, texture_orig2t = get_high_res_texture(
        500,
        minx2, minx2 + edge_size2, miny2, miny2 + edge_size2, model_F_atlas, device
    )
    checkerboard_ = np.array(Image.open(str("checkerboard.png"))).astype(np.float64) / 255.
    checkerboard_ = cv2.resize(checkerboard_, (500, 500))
    checkerboard_ = torch.from_numpy(checkerboard_[:, :, :3])

    checkerboard = (checkerboard_ * 0.3 + texture_orig1t * 0.7)
    checkerboard2 = (checkerboard_ * 0.3 + texture_orig2t * 0.7)

    alpha_reconstruction = np.zeros((resy, resx, number_of_frames))
    video_frames_reconstruction_edit1 = np.zeros((resy, resx, 3, number_of_frames))
    video_frames_reconstruction_edit1_checkerboard = np.ones((resy, resx, 3, number_of_frames)) * 0.5
    video_frames_reconstruction_edit2 = np.zeros((resy, resx, 3, number_of_frames))
    video_frames_reconstruction_edit2_checkerboard = np.ones((resy, resx, 3, number_of_frames)) * 0.5
    video_frames_reconstruction_edit = np.zeros((resy, resx, 3, number_of_frames))

    video_frames_reconstruction = np.zeros((resy, resx, 3, number_of_frames))
    masks1 = np.zeros((edited_tex1.shape[0], edited_tex1.shape[1]))
    masks2 = np.zeros((edited_tex2.shape[0], edited_tex2.shape[1]))

    flow_loss1_video = np.zeros((resy, resx, number_of_frames))
    flow_loss2_video = np.zeros((resy, resx, number_of_frames))
    flow_alpha_loss_video = np.zeros((resy, resx, number_of_frames))

    rigidity_loss1_video = np.zeros((resy, resx, number_of_frames))
    rigidity_loss2_video = np.zeros((resy, resx, number_of_frames))
    rgb_error_video = np.zeros((resy, resx, number_of_frames))
    rgb_residual_video = np.zeros((resy, resx, 3, number_of_frames))

    uv1_frames_reconstruction = np.zeros((resy, resx, 3, number_of_frames))
    uv2_frames_reconstruction = np.zeros((resy, resx, 3, number_of_frames))

    all_masks1 = np.zeros((1000, 1000, number_of_frames))
    all_masks2 = np.zeros((1000, 1000, number_of_frames))

    with torch.no_grad():
        for f in range(number_of_frames):
            print(f)

            # 전체 y좌표, 전체 x좌표
            relis_i, reljs_i = torch.where(torch.ones(resy, resx) > 0)

            # split the coordinates of the entire image
            # such that no more than 100k coordinates in each batch
            relisa = np.array_split(relis_i.numpy(), np.ceil(relis_i.shape[0] / 100000))
            reljsa = np.array_split(reljs_i.numpy(), np.ceil(relis_i.shape[0] / 100000))

            m1_alpha_v = []
            m1_alpha_u = []
            m1_alpha_alpha = []

            m2_alpha_v = []
            m2_alpha_u = []
            m2_alpha_alpha = []

            for i in range(len(relisa)):
                # 좌표를 normalize 해준다.
                relis = torch.from_numpy(relisa[i]).unsqueeze(1) / (larger_dim / 2) - 1
                reljs = torch.from_numpy(reljsa[i]).unsqueeze(1) / (larger_dim / 2) - 1
                # Map video indices to uv_coordinates using the two mapping networks:
                uv_temp1 = model_F_mapping1(
                    torch.cat((reljs, relis,
                               (f / (number_of_frames / 2.0) - 1) * torch.ones_like(relis)),
                              dim=1).to(device))
                uv_temp2 = model_F_mapping2(
                    torch.cat((reljs, relis,
                               (f / (number_of_frames / 2.0) - 1) * torch.ones_like(relis)),
                              dim=1).to(device))
                # Sample RGB values from the atlas:
                rgb_current1 = model_F_atlas(uv_temp1 * 0.5 + 0.5)
                rgb_current2 = model_F_atlas(uv_temp2 * 0.5 - 0.5)

                rgb_current1 = (rgb_current1 + 1) * 0.5
                rgb_current2 = (rgb_current2 + 1) * 0.5

                alpha = 0.5 * (model_alpha(torch.cat((reljs, relis,
                                                      (f / (number_of_frames / 2.0) - 1) * torch.ones_like(relis)),
                                                     dim=1).to(device)) + 1.0)
                alpha = alpha * 0.99
                alpha = alpha + 0.001

                # pixels reconstruction from the MLPs:
                rgb_current = rgb_current1 * alpha + rgb_current2 * (1.0 - alpha)

                jif_foreground = torch.cat((torch.from_numpy(reljsa[i]).unsqueeze(-1),
                                            torch.from_numpy(relisa[i]).unsqueeze(-1),
                                            torch.ones_like(torch.from_numpy(relisa[i]).unsqueeze(-1)) * f),
                                           dim=1).T.unsqueeze(-1)

                # 여기까지 데이터를 모델에 거쳐 픽셀 복원을 했다.
                # 이 다음부터는 loss 계산을 해서 시각화하는 방식으로 모델을 평가한다.

                # reconstruct rigidity losses for visualization:
                rigidity_loss1 = get_rigidity_loss(
                    jif_foreground,
                    derivative_amount,
                    larger_dim,
                    number_of_frames,
                    model_F_mapping1,
                    uv_temp1, device,
                    uv_mapping_scale= uv_mapping_scale,
                    return_all=True)

                rigidity_loss2 = get_rigidity_loss(
                    jif_foreground,
                    derivative_amount,
                    larger_dim,
                    number_of_frames,
                    model_F_mapping2,
                    uv_temp2, device,
                    uv_mapping_scale=uv_mapping_scale,
                    return_all=True)

                # Reconstruct flow losses for visualization:
                if f < (number_of_frames - 1):
                    flow_loss1 = get_optical_flow_loss_all(
                        jif_foreground, uv_temp1, larger_dim,
                        number_of_frames, model_F_mapping1, optical_flows, optical_flows_mask,
                        uv_mapping_scale, device, alpha=alpha)
                    flow_loss2 = get_optical_flow_loss_all(
                        jif_foreground, uv_temp2, larger_dim,
                        number_of_frames, model_F_mapping2, optical_flows, optical_flows_mask,
                        uv_mapping_scale, device, alpha=1-alpha)
                else: # for not calculating the optical flow between the last frame and the next non-existing frame
                    flow_loss1 = torch.zeros_like(relis).squeeze()
                    flow_loss2 = torch.zeros_like(relis).squeeze()

                flow_alpha_loss = get_optical_flow_alpha_loss_all(model_alpha,
                                                                  jif_foreground,
                                                                  alpha,
                                                                  larger_dim,
                                                                  number_of_frames, optical_flows,
                                                                  optical_flows_mask, device)
                # Same uv_values from each frame for visualization:
                uv_temp1 = uv_temp1.detach().cpu()
                uv_temp2 = uv_temp2.detach().cpu()

                uv1_frames_reconstruction[relisa[i], reljsa[i], 0, f] = uv_temp1[:, 0]
                uv1_frames_reconstruction[relisa[i], reljsa[i], 1, f] = uv_temp1[:, 1]

                uv2_frames_reconstruction[relisa[i], reljsa[i], 0, f] = uv_temp2[:, 0]
                uv2_frames_reconstruction[relisa[i], reljsa[i], 1, f] = uv_temp2[:, 1]

                # pixels reconstruction from the edited foreground texture:
                rgb21, pointsx1, pointsy1, relevant1 = get_colors(1000,
                                                                  minx, minx + edge_size, miny, miny + edge_size,
                                                                  uv_temp1[:, 0] * 0.5 + 0.5,
                                                                  uv_temp1[:, 1] * 0.5 + 0.5, edited_tex1)
                # reconstruct background pixels from the edited background
                rgb22, pointsx2, pointsy2, relevant2 = get_colors(1000,
                                                                  minx2, minx2 + edge_size2, miny2,
                                                                  miny2 + edge_size2,
                                                                  uv_temp2[:, 0] * 0.5 - 0.5,
                                                                  uv_temp2[:, 1] * 0.5 - 0.5,
                                                                  edited_tex2)
                # pixels reconstruction from the checkerboard texture:
                rgb21_tex, pointsx1_tex, pointsy1_tex, relevant1_tex = get_colors(500,
                                                                  minxt, minxt + edge_sizet, minyt,
                                                                  minyt + edge_sizet,
                                                                  uv_temp1[:, 0] * 0.5 + 0.5,
                                                                  uv_temp1[:, 1] * 0.5 + 0.5,
                                                                  checkerboard)

                rgb22_tex, pointsx2_tex, pointsy2_tex, relevant2_tex = get_colors(500,
                                                                  minx2, minx2 + edge_size2, miny2,
                                                                  miny2 + edge_size2,
                                                                  uv_temp2[:, 0] * 0.5 - 0.5,
                                                                  uv_temp2[:, 1] * 0.5 - 0.5,
                                                                  checkerboard2)

                m1_alpha_v.append(pointsy1)
                m1_alpha_u.append(pointsx1)
                m1_alpha_alpha.append(alpha.cpu().squeeze()[relevant1].numpy())

                m2_alpha_v.append(pointsy2)
                m2_alpha_u.append(pointsx2)
                m2_alpha_alpha.append(1.0 - alpha.cpu().squeeze()[relevant2].numpy())

                # define the mask of the foreground texture
                # using the mapped alpha value of the current frames
                # (if a foreground atlas pixel was ever used, set the mask to its alpha)
                try:
                    masks1[np.ceil(pointsy1).astype((np.int64)), np.ceil(pointsx1).astype((np.int64))] = np.maximum(
                        masks1[np.ceil(pointsy1).astype((np.int64)), np.ceil(pointsx1).astype((np.int64))],
                        alpha.cpu().squeeze()[relevant1].numpy())
                    masks1[np.floor(pointsy1).astype((np.int64)), np.floor(pointsx1).astype((np.int64))] = np.maximum(
                        masks1[np.floor(pointsy1).astype((np.int64)), np.floor(pointsx1).astype((np.int64))],
                        alpha.cpu().squeeze()[relevant1].numpy())
                    masks1[np.floor(pointsy1).astype((np.int64)), np.ceil(pointsx1).astype((np.int64))] = np.maximum(
                        masks1[np.floor(pointsy1).astype((np.int64)), np.ceil(pointsx1).astype((np.int64))],
                        alpha.cpu().squeeze()[relevant1].numpy())
                    masks1[np.ceil(pointsy1).astype((np.int64)), np.floor(pointsx1).astype((np.int64))] = np.maximum(
                        masks1[np.ceil(pointsy1).astype((np.int64)), np.floor(pointsx1).astype((np.int64))],
                        alpha.cpu().squeeze()[relevant1].numpy())
                except Exception:
                    pass

                # relevant 조건에 맞는 좌표에 RGB값을 할당한다.
                # 이 때 rgb21은 foreground 텍스처에서 추출한 값이고
                #      rgb22는 background 텍스처에서 추출한 값이다.
                video_frames_reconstruction_edit1[relisa[i][relevant1], reljsa[i][relevant1], :, f] = rgb21 * \
                                                                                                      alpha.cpu().numpy()[
                                                                                                          relevant1]
                video_frames_reconstruction_edit2[relisa[i][relevant2], reljsa[i][relevant2], :, f] = rgb22

                video_frames_reconstruction_edit[relisa[i][relevant1], reljsa[i][relevant1], :,
                f] = video_frames_reconstruction_edit[relisa[i][relevant1], reljsa[i][relevant1], :, f] + rgb21 * \
                     alpha.cpu().numpy()[relevant1]
                video_frames_reconstruction_edit[relisa[i][relevant2], reljsa[i][relevant2], :,
                f] = video_frames_reconstruction_edit[relisa[i][relevant2], reljsa[i][relevant2], :, f] + rgb22 * \
                     (1 - alpha).cpu().numpy()[relevant2]

                video_frames_reconstruction[relisa[i], reljsa[i], :, f] = rgb_current.detach().cpu(
                ).numpy()
                alpha_reconstruction[relisa[i], reljsa[i], f] = alpha[:, 0].detach().cpu(
                ).numpy()
                flow_loss1_video[relisa[i], reljsa[i], f] = flow_loss1.cpu().numpy()
                flow_loss2_video[relisa[i], reljsa[i], f] = flow_loss2.cpu().numpy()
                flow_alpha_loss_video[relisa[i], reljsa[i], f] = flow_alpha_loss.cpu().squeeze().numpy()
                rigidity_loss1_video[relisa[i], reljsa[i], f] = rigidity_loss1.cpu().numpy()
                rigidity_loss2_video[relisa[i], reljsa[i], f] = rigidity_loss2.cpu().numpy()
                rgb_error_video[relisa[i], reljsa[i], f] = (
                        (video_frames[relisa[i], reljsa[i], :, f] - rgb_current.cpu()).norm(dim=1) ** 2).numpy()
                rgb_residual_video[relisa[i], reljsa[i], :, f] = (
                    (video_frames[relisa[i], reljsa[i], :, f] - rgb_current.cpu())).numpy()
                video_frames_reconstruction_edit1_checkerboard[relisa[i][relevant1_tex], reljsa[i][relevant1_tex], :,
                f] = video_frames_reconstruction_edit1_checkerboard[relisa[i][relevant1_tex], reljsa[i][relevant1_tex],
                     :,
                     f] * (1 - alpha).cpu().numpy()[relevant1_tex] + rgb21_tex * alpha.cpu().numpy()[relevant1_tex]

                video_frames_reconstruction_edit2_checkerboard[relisa[i][relevant2_tex], reljsa[i][relevant2_tex], :,
                f] = rgb22_tex
            if show_atlas_alpha: #this part is slow, not running during training
                cur_mask1 = interpolate_alpha(m1_alpha_v, m1_alpha_u, m1_alpha_alpha)
                cur_mask2 = interpolate_alpha(m2_alpha_v, m2_alpha_u, m2_alpha_alpha)
                all_masks1[:, :, f] = cur_mask1
                all_masks2[:, :, f] = cur_mask2
    if show_atlas_alpha: # this part is slow, not running during training
        all_masks1[np.isnan(all_masks1)] = 0
        all_masks2[np.isnan(all_masks2)] = 0
        masks1_alpha = np.nanmedian(all_masks1[:, :, :], axis=2)
        masks2_alpha = np.nanpercentile(all_masks2[:, :, :], 90, axis=2)

    uv1_frames_reconstruction = normalize_uv_images(uv1_frames_reconstruction, 0.5, edge_size, minx, miny)
    uv2_frames_reconstruction = normalize_uv_images(uv2_frames_reconstruction, -0.5, edge_size2, minx2, miny2)

    Path(evaluation_folder).mkdir(parents=True, exist_ok=True)
    mpimg.imsave("%s/texture_edit1.png" % (evaluation_folder),
                     (masks1[:, :, np.newaxis] * edited_tex1.numpy() * 255).astype(np.uint8))
    mpimg.imsave("%s/texture_orig1.png" % (evaluation_folder),
                     (masks1[:, :, np.newaxis] * texture_orig1.numpy() * 255).astype(np.uint8))

    mpimg.imsave("%s/texture_edit2.png" % (evaluation_folder),
                     (masks2[:, :, np.newaxis] * edited_tex2.numpy() * 255).astype(np.uint8))

    if show_atlas_alpha:
        mpimg.imsave("%s/texture_orig1_alpha.png" % (evaluation_folder),
                         (np.concatenate((texture_orig1.numpy(), masks1_alpha[:, :, np.newaxis]), axis=2) * (255)).astype(
                         np.uint8))
        mpimg.imsave("%s/texture_orig2_alpha.png" % (evaluation_folder),
                         (np.concatenate((texture_orig2.numpy(), masks2_alpha[:, :, np.newaxis]), axis=2) * (255)).astype(
                             np.uint8))

    mpimg.imsave("%s/texture_orig2.png" % (evaluation_folder),
                     (masks2[:, :, np.newaxis] * texture_orig2.numpy() * 255).astype(np.uint8))

    writer_t_edited1 = imageio.get_writer(
            "%s/edited1_%s.mp4" % (evaluation_folder, vid_name), fps=10)
    writer_t_edited1_tex = imageio.get_writer(
            "%s/edited1_tex_%s.mp4" % (evaluation_folder, vid_name), fps=10)
    writer_t_edited2_tex = imageio.get_writer(
            "%s/edited2_%s.mp4" % (evaluation_folder, vid_name), fps=10)

    writer_t_edited2 = imageio.get_writer(
            "%s/edited2_%s.mp4" % (evaluation_folder, vid_name), fps=10)
    writer_alpha = imageio.get_writer(
            "%s/alpha_%s.mp4" % (evaluation_folder, vid_name), fps=10)

    writer_im_rec = imageio.get_writer(
            "%s/reconstruction_%s.mp4" % (evaluation_folder, vid_name), fps=10)

    writer_residuals = imageio.get_writer(
            "%s/residuals_%s.mp4" % (evaluation_folder, vid_name), fps=10)

    writer_alpha_vs_mask_rcnn = imageio.get_writer(
            "%s/alpha_vs_mask_rcnn_%s.mp4" % (evaluation_folder, vid_name), fps=10)

    writer_edit = imageio.get_writer(
            "%s/edit_%s.mp4" % (evaluation_folder, vid_name), fps=10)

    writer_uv_1 = imageio.get_writer(
            "%s/uv_1_%s.mp4" % (evaluation_folder, vid_name),
            fps=10)

    writer_uv_1_masked = imageio.get_writer(
            "%s/uv_1_masked_%s.mp4" % (evaluation_folder, vid_name),
            fps=10)

    writer_uv_2 = imageio.get_writer(
            "%s/uv_2_%s.mp4" % (evaluation_folder, vid_name),
            fps=10)
    writer_checkerboard_1 = imageio.get_writer(
            "%s/checkerboard_1_%s.mp4" % (evaluation_folder, vid_name),
            fps=10)
    writer_checkerboard_2 = imageio.get_writer(
            "%s/checkerboard_2_%s.mp4" % (evaluation_folder, vid_name),
            fps=10)

    writer_global_info = imageio.get_writer(
            "%s/global_info_%s.mp4" % (evaluation_folder, vid_name), fps=10)

    pnsrs = np.zeros((number_of_frames, 1))
    # save evaluation videos:
    for i in range(number_of_frames):
        print(i)

        cur = (video_frames_reconstruction_edit1[:, :, :, i])
        cc = np.concatenate(
                (cur, cv2.resize(masks1[:, :, np.newaxis] * edited_tex1.numpy(),
                                 (cur.shape[0], cur.shape[0]))), axis=1)
        writer_t_edited1.append_data((cc * 255).astype(np.uint8))

        cur = (video_frames_reconstruction_edit1_checkerboard[:, :, :, i])
        cc = np.concatenate(
                (cur, cv2.resize(checkerboard.numpy(), (cur.shape[0], cur.shape[0]))), axis=1)
        writer_t_edited1_tex.append_data((cc * 255).astype(np.uint8))
        writer_checkerboard_1.append_data((cur * 255).astype(np.uint8))

        cur = (video_frames_reconstruction_edit2_checkerboard[:, :, :, i])
        cc = np.concatenate(
                (cur, cv2.resize(checkerboard2.numpy(), (cur.shape[0], cur.shape[0]))),
                axis=1)
        writer_t_edited2_tex.append_data((cc * (255)).astype(np.uint8))
        writer_checkerboard_2.append_data((cur * 255).astype(np.uint8))

        writer_alpha.append_data((alpha_reconstruction[:, :, i] * 255).astype(np.uint8))

        cur = (video_frames_reconstruction_edit2[:, :, :, i])
        cc = np.concatenate(
                (cur, cv2.resize(masks2[:, :, np.newaxis] * edited_tex2.numpy(),
                                 (cur.shape[0], cur.shape[0]))), axis=1)
        writer_t_edited2.append_data((cc * 255).astype(np.uint8))

        alpha_vs_mask_rcnn_cur = np.transpose(np.stack(
                (mask_frames[:, :, i].numpy(), alpha_reconstruction[:, :, i], np.zeros_like(mask_frames[:, :, i].numpy()))),
                (1,2,0))

        writer_alpha_vs_mask_rcnn.append_data((alpha_vs_mask_rcnn_cur * 255.0).astype(np.uint8))
        writer_edit.append_data((video_frames_reconstruction_edit[:, :, :, i] * (255)).astype(np.uint8))
        writer_im_rec.append_data((video_frames_reconstruction[:, :, :, i] * (255)).astype(np.uint8))
        writer_residuals.append_data(((rgb_residual_video[:, :, :, i] + 0.5) * 255).astype(np.uint8))

        writer_uv_1.append_data((uv1_frames_reconstruction[:, :, :, i] * (255)).astype(np.uint8))
        writer_uv_2.append_data((uv2_frames_reconstruction[:, :, :, i] * (255)).astype(np.uint8))
        writer_uv_1_masked.append_data(
                (uv1_frames_reconstruction[:, :, :, i] * alpha_reconstruction[:, :, i][:, :, np.newaxis] * (
                    255)).astype(
                    np.uint8))

        pnsrs[i] = skimage.metrics.peak_signal_noise_ratio(
                video_frames[:, :, :, i].numpy(),
                video_frames_reconstruction[:, :, :, i],
                data_range=1)

        fig = plt.figure(figsize=(20, 10))
        plt.subplot(3, 4, 3)
        plt.imshow(rgb_error_video[:, :, i], vmin=0.0, vmax=0.2)
        plt.colorbar()
        plt.title("RGB error")

        plt.subplot(3, 4, 12)
        plt.imshow(rigidity_loss1_video[:, :, i], vmin=2.8, vmax=50.0)
        plt.colorbar()
        plt.title("rigidity_loss1")

        plt.subplot(3, 4, 7)
        plt.imshow(flow_alpha_loss_video[:, :, i], vmin=0, vmax=1)
        plt.colorbar()
        plt.title("flow_alpha_loss")

        plt.subplot(3, 4, 11)
        plt.imshow(rigidity_loss2_video[:, :, i], vmin=2.8, vmax=50.0)
        plt.colorbar()
        plt.title("rigidity_loss2")

        plt.subplot(3, 4, 9)
        plt.imshow(flow_loss1_video[:, :, i], vmin=0.0, vmax=2.0)
        plt.colorbar()
        plt.title("flow_loss1")

        plt.subplot(3, 4, 10)
        plt.imshow(flow_loss2_video[:, :, i], vmin=0.0, vmax=2.0)
        plt.colorbar()
        plt.title("flow_loss2")

        plt.subplot(3, 4, 5)
        plt.imshow(alpha_reconstruction[:, :, i], vmin=0.0, vmax=1.0)
        plt.colorbar()
        plt.title("alpha")
        plt.subplot(3, 4, 6)
        plt.imshow(alpha_vs_mask_rcnn_cur, vmin=0.0, vmax=1.0)
        plt.colorbar()
        plt.title("alpha_vs_mask_rcnn")

        plt.subplot(3, 4, 1)
        plt.imshow(video_frames_reconstruction[:, :, :, i], vmin=0.0, vmax=1.0)
        plt.colorbar()
        plt.title("video_reconstruction")

        plt.subplot(3, 4, 2)
        plt.imshow(video_frames[:, :, :, i].numpy(), vmin=0.0, vmax=1.0)
        plt.colorbar()
        plt.title("original_video")

        imm = get_img_from_fig(fig)
        writer_global_info.append_data(imm)
        plt.close(fig)

    print(pnsrs.mean())
    writer_t_edited2_tex.close()
    writer_t_edited1_tex.close()
    writer_t_edited1.close()
    writer_t_edited2.close()
    writer_im_rec.close()
    writer_alpha.close()
    writer_alpha_vs_mask_rcnn.close()
    writer_global_info.close()
    writer_residuals.close()
    writer_edit.close()
    writer_uv_1.close()
    writer_uv_2.close()
    writer_checkerboard_1.close()
    writer_checkerboard_2.close()
    writer_uv_1_masked.close()

    # save the psnr result as the name of a dummy file
    file1 = open('%s/%06d/PSNR_%f' % (results_folder, iteration, pnsrs.mean()), "a")
    file1.close()
    if save_checkpoint:
        writer.add_image("Train/atlas1", masks1[:, :, np.newaxis] * texture_orig1.numpy(), iteration, dataformats="HWC")
        writer.add_image("Train/atlas2", masks2[:, :, np.newaxis] * texture_orig2.numpy(), iteration, dataformats="HWC")
        writer.add_image(
            "Train/recon_frame_0",
            video_frames_reconstruction[:, :, :, 0],
            iteration,
            dataformats='HWC')
        writer.add_image(
            "Train/recon_frame_end",
            video_frames_reconstruction[:, :, :, -1],
            iteration,
            dataformats='HWC')

    return alpha_reconstruction
