import torch


'''
논문에서 Eq.7에 해당하는 gradient loss를 계산하는 역할을 한다.
논문의 Reconstruction Loss는 "RGB loss"와 지금 구현하는 "gradient loss"의 합인데,
RGB Loss는 학습을 진행하는 코드에서 바로 구현하기 때문에 이 파일에서는 빠져 있다.

<입력값에 대해> 
resx, resy : config 파일에서 각각 768, 432로 정해놓았다.
-> 해상도 x, 해상도 y일 것으로 추정.

model_f_mapping1 : foreground atlas 전용 u,v를 추론하는 layer
model_f_mapping2 : background atlas 전용 u,v를 추론하는 layer
model_F_atlas : u,v를 바탕으로 atlas의 색을 추론하는 layer

video_frames_dx : torch.zeros((resy, resx, 3, number_of_frames)) 로 정해놓았다. 
                  -> 해상도 y, 해상도 x, RGB, 프레임 번호
video_frames_dy : torch.zeros((resy, resx, 3, number_of_frames)) 로 정해놓았다. 
                  -> 해상도 y, 해상도 x, RGB, 프레임 번호
                  
jif_current : 정체불명이다. 이게 뭐지?
-> (추정) j와 i는 iterate하는 index이고 f는 frame_num을 의미하는 것 같다.
-> 즉, 현재 j,i 좌표와 현재 프레임을 뜻하는 것.
-> j 좌표는 resx에 해당하고, i 좌표는 resy에 해당할 것으로 보임.

jif_foreground를 가상으로 만들어서 shape를 확인해보니 [3,82944,1]이다. 무슨 의미일까.
jif_all을 가상으로 만들어서 shape를 확인해보니 [3, 23131425]이다.
jif_current를 가상으로 만들어서 shape를 확인해보니 [3, 10000, 1]이다.

하나하나 확인해보니 다음과 같다는 결론을 내렸다.
첫 번째 3은 각각 resx, resy, frame_num을 가리킨다.
두번째 숫자는 한 batch에 포함되는 수를 나타낸다.
세번째 숫자는 의미없는 차원이다. (squeeze 해서 쓸 수 있다.)

'''

def get_gradient_loss(video_frames_dx, video_frames_dy, jif_current,
                      model_F_mapping1, model_F_mapping2, model_F_atlas,
                      rgb_output_foreground, device, resx, number_of_frames,
                      model_alpha):

    # 이렇게 읽으면 된다.
    # x+1, y, t
    xplus1yt_foreground = torch.cat(
        ((jif_current[0, :] + 1) / (resx / 2) - 1, jif_current[1, :] / (resx / 2) - 1,
         jif_current[2, :] / (number_of_frames / 2.0) -1),
        dim=1).to(device)

    # x, y+1, t
    xyplus1t_foreground = torch.cat(
        ((jif_current[0, :]) / (resx / 2) - 1,
         (jif_current[1, :] + 1) / (resx / 2) - 1,
         jif_current[2, :] / (number_of_frames / 2.0) - 1),
        dim=1).to(device)

    # alpha, x+1
    alphaxplus1 = 0.5 * (model_alpha(xplus1yt_foreground) + 1.0)
    alphaxplus1 = alphaxplus1 * 0.99
    alphaxplus1 = alphaxplus1 + 0.001

    # alpha, y+1
    alphayplus1 = 0.5 * (model_alpha(xyplus1t_foreground) + 1.0)
    alphayplus1 = alphayplus1 * 0.99
    alphayplus1 = alphayplus1 + 0.001

    # 미리 계산해 놓은 x,y 방향에 대한 discrete derivative
    # python list(array) 특성상 y축(세로축)이 첫번째(i), x축(가로축)이 두번째(j)임을 기억하자.
    #
    # squeeze 함수는 차원이 1인 차원을 제거해준다. 따로 차원을 설정하지 않으면 1인 차원을 모두 제거한다.
    # 차원을 설정해주면 그 차원만 제거한다.
    #
    # pytorch에서 x라는 텐서가 있다고 하자.
    # 이 때 x[1, :]는 x[1]과 같은 결과를 낸다.
    # 굳이 어려워 보이게 왜 이렇게 하냐면
    # 콜론이 인덱스들의 중간에 있을 때는 다른 결과를 내기 때문에 일종의 통일성을 주기 위해서인 것 같다.
    #
    # video_frames와 video_frames_dx, video_frames_dy는
    # 모두 (resy, resx, 3, number_of_frames)의 shape을 가지고 있다.
    # 여기서 3은 RGB이다.
    rgb_dx_gt = video_frames_dx[jif_current[1, :], jif_current[0, :], :,
                jif_current[2, :]].squeeze(1).to(device)
    rgb_dy_gt = video_frames_dy[jif_current[1, :], jif_current[0, :], :,
                jif_current[2, :]].squeeze(1).to(device)

    # offset이 각각 1픽셀인 원본 위치에 대한 uv 좌표
    # background
    uv_foreground2_xyplus1t = model_F_mapping2(xyplus1t_foreground)
    uv_foreground2_xplus1yt = model_F_mapping2(xplus1yt_foreground)
    # foreground
    uv_foreground1_xyplus1t = model_F_mapping1(xyplus1t_foreground)
    uv_foreground1_xplus1yt = model_F_mapping1(xplus1yt_foreground)

    # offset이 각각 1픽셀인 원본 위치에 대한 atlas RGB 값 (foreground / background)
    rgb_output1_xyplus1t = (model_F_atlas(uv_foreground1_xyplus1t * 0.5 + 0.5) + 1.0) * 0.5
    rgb_output1_xplus1yt = (model_F_atlas(uv_foreground1_xplus1yt * 0.5 + 0.5) + 1.0) * 0.5
    rgb_output2_xyplus1t = (model_F_atlas(uv_foreground2_xyplus1t * 0.5 + 0.5) + 1.0) * 0.5
    rgb_output2_xplus1yt = (model_F_atlas(uv_foreground2_xplus1yt * 0.5 + 0.5) + 1.0) * 0.5

    # 복원한 RGB 값 (Reconstructed RGB values)
    rgb_output_foreground_xyplus1t = rgb_output1_xyplus1t * alphayplus1 + rgb_output2_xyplus1t * (1.0 - alphayplus1)
    rgb_output_foreground_xplus1yt = rgb_output1_xplus1yt * alphaxplus1 + rgb_output2_xplus1yt * (1.0 - alphaxplus1)

    # 복원한 RGB 값을 미분값을 계산하는데 사용한다.
    rgb_dx_output = rgb_output_foreground_xplus1yt - rgb_output_foreground
    rgb_dy_output = rgb_output_foreground_xyplus1t - rgb_output_foreground

    gradient_loss = torch.mean(
        (rgb_dx_gt - rgb_dx_output).norm(dim=1) ** 2 + (rgb_dy_gt - rgb_dy_output).norm(dim=1) ** 2)
    return gradient_loss


'''
논문에서 rigidity loss에 해당하는 구현.
'''

def get_rigidity_loss(jif_foreground, derivative_amount, resx,
                      number_of_frames, model_F_mapping, uv_foreground,
                      device, uv_mapping_scale=1.0, return_all=False):
    # (x, y-derivative_amount, t)와 (x-derivative_amount, y, t)를 붙여 xyt_p를 구한다.
    # 각각을 구할 때 값이 [-1, 1] 사이값을 가지도록 normalize 해준다.
    # y좌표
    is_patch = torch.cat((jif_foreground[1, :] - derivative_amount, jif_foreground[1, :])) / (resx / 2) - 1
    # x좌표
    js_patch = torch.cat((jif_foreground[0, :], jif_foreground[0, :] - derivative_amount)) / (resx / 2) - 1
    # frame
    fs_patch = torch.cat((jif_foreground[2, :], jif_foreground[2, :])) / (number_of_frames / 2.0) - 1
    xyt_p = torch.cat((js_patch, is_patch,fs_patch), dim=1).to(device)

    uv_p = model_F_mapping(xyt_p)
    # u_p[0,:]= u(x,y-derivative_amount,t).
    # u_p[1,:]= u(x-derivative_amount,y,t)
    u_p = uv_p[:, 0].view(2, -1)
    # v_p[0,:]= u(x,y-derivative_amount,t).
    # v_p[1,:]= v(x-derivative_amount,y,t)
    v_p = uv_p[:, 1].view(2, -1)

    # unsqueeze 함수는 squeeze 함수의 반대로 1인 차원을 생성하는 함수다.
    # 그래서 어느 차원에 1인 차원을 생성할 지 꼭 지정해주어야 한다.

    # u_p_d_[0,:]=u(x,y,t)-u(x,y-derivative_amount,t)
    # u_p_d_[1,:]= u(x,y,t)-u(x-derivative_amount,y,t).
    u_p_d_ = uv_foreground[:, 0].unsqueeze(0) - u_p
    # v_p_d_[0,:]=v(x,y,t)-v(x,y-derivative_amount,t).
    # v_p_d_[1,:]= v(x,y,t)-v(x-derivative_amount,y,t).
    v_p_d_ = uv_foreground[:, 1].unsqueeze(0) - v_p

    # 단위를 맞추기 위한 코드 : uv 좌표에서 1은 이미지 공간에서 resx/2를 의미한다.
    du_dx = u_p_d_[1, :] * resx / 2
    du_dy = u_p_d_[0, :] * resx / 2
    dv_dx = v_p_d_[1, :] * resx / 2
    dv_dy = v_p_d_[0, :] * resx / 2

    # torch.cat 함수는 텐서를 concatenate 한다.
    # 차원을 지정하면 그 차원으로 두 텐서의 차원을 더하는 방향으로 concatenate 한다.
    # 차원을 지정하지 않았을 때 기본값은 dim=0이다.
    jacobians = torch.cat((torch.cat((du_dx.unsqueeze(-1).unsqueeze(-1), du_dy.unsqueeze(-1).unsqueeze(-1)), dim=2),
                           torch.cat((dv_dx.unsqueeze(-1).unsqueeze(-1), dv_dy.unsqueeze(-1).unsqueeze(-1)),
                                     dim=2)),
                          dim=1)
    jacobians = jacobians / uv_mapping_scale
    jacobians = jacobians / derivative_amount

    # Jacobian이 최대한 회전변환 행렬이 되도록 제한하는 loss를 적용한다.
    JtJ = torch.matmul(jacobians.transpose(1,2), jacobians)

    a = JtJ[:, 0, 0] + 0.001
    b = JtJ[:, 0, 1]
    c = JtJ[:, 1, 0]
    d = JtJ[:, 1, 1] + 0.001

    JTJinv = torch.zeros_like(jacobians).to(device)
    JTJinv[:, 0, 0] = d
    JTJinv[:, 0, 1] = -b
    JTJinv[:, 1, 0] = -c
    JTJinv[:, 1, 1] = a
    JTJinv = JTJinv / ((a * d - b * c).unsqueeze(-1).unsqueeze(-1))

    # 논문에서 Eq.9를 볼 것
    rigidity_loss = (JtJ ** 2).sum(1).sum(1).sqrt() + (JTJinv ** 2).sum(1).sum(1).sqrt()

    if return_all:
        return rigidity_loss
    else:
        return rigidity_loss.mean()

'''
모든 픽셀에 대해 optical flow loss(논문에서 Eq. 11)를 계산한다.
다만 평균값을 취하지는 않는다.
이 함수는 loss를 시각화 할 때 적절하다.

Optical flow를 추정하기 위해서는 현실을 훼손하지 않는 범위 내에서 적절하게 가정을 세운다.
전제조건 두가지는 다음과 같다.
- 1. color/brightness constancy : 어떤 픽셀과 그 픽셀의 주변 픽셀의 색/밝기는 같음을 가정한다.
- 2. small motion : Frame 간 움직임이 작아서 어떤 픽셀 점이 멀리 움직이지 않는 것을 가정한다.

<입력값에 대해>
다른 함수들에서 자세히 설명해 놓았다.
'''
def get_optical_flow_loss_all(jif_foreground, uv_foreground,
                              resx, number_of_frames, model_F_mapping,
                              optical_flows, optical_flows_mask, uv_mapping_scale, device,
                              alpha=1.0):
    xyt_foreground_forward_should_match, relevant_batch_indices_forward = get_corresponding_flow_matches_all(
        jif_foreground, optical_flows_mask, optical_flows, resx, number_of_frames)
    uv_foreground_forward_should_match = model_F_mapping(xyt_foreground_forward_should_match.to(device))

    errors = (uv_foreground_forward_should_match - uv_foreground).norm(dim=1)
    errors[relevant_batch_indices_forward == False] = 0
    errors = errors * (alpha.squeeze())

    return errors * resx / (2 * uv_mapping_scale)


'''
논문에서 Eq. 11로 표시한 optical flow loss를 계산한다.
jif_foreground : shape ([3, 10000, 1])
                 즉, ([좌표와 프레임, samples, 1])
                 좌표와 프레임 쌍이 samples 만큼 있다.
                 
uv_foreground : shape ([10000,2]) 
                즉, ([samples, 좌표])
                u,v좌표가 samples 갯수만큼 있다.
                
                uv_foreground는 (train 파일에서) 
                jif_foreground(jif_current) 변수를 xyt_current 변수로 변환시킨 다음
                mapping_MLP를 거친 것이다.
                
                여기서 xyt_current 변수는 jif_foreground(jif_current)가
                [-1, 1] 사이의 값을 갖도록 normalize 해준 것이다.

optical_flows_reverse : shape ([resy, resx, 2, number_of_frames, 1])
                        optical flows를 역으로 (t+1 프레임에서 t 프레임으로) 계산한 것
                        
                        아래에서도 설명하겠지만
                        세번째 차원 "2"는 픽셀의 움직임을 
                        horizontal flow, vertical flow로 표현한 것이다.

optical_flows_reverse_mask: 
                        shape ([resy, resx, number_of_frames, 1])
                        
optical_flows :         shape ([resy, resx, 2, number_of_frames, 1])
                        optical flows를 (t 프레임에서 t+1 프레임으로) 계산한 것
                        
                        그런데, 세번째 차원 "2"는 무엇을 뜻하는걸까?
                        -> Horizontal flow, Vertical flow.
                        optical flow는 Optical Field를 구하기 위해 이전 프레임과 현재 프레임의
                        차이를 이용하고 픽셀값과 주변 픽셀들과의 관계를 통해 각 픽셀의 이동을 계산해서
                        추출한다. 
                        (내 의견) 이 때 픽셀이 움직이는 것은 벡터로써, 수평적, 수직적 component로
                        분해할 수 있는데 이 components를 표현한 것인 듯 하다.

optical_flows_mask:     shape ([resy, resx, number_of_frames, 1])
'''
def get_optical_flow_loss(jif_foreground, uv_foreground, optical_flows_reverse,
                          optical_flows_reverse_mask, resx, number_of_frames,
                          model_F_mapping, optical_flows, optical_flows_mask,
                          uv_mapping_scale, device, use_alpha=False, alpha=1.0):
    # Forward flow:
    uv_foreground_forward_relevant, xyt_foreground_forward_should_match, relevant_batch_indices_forward = get_corresponding_flow_matches(
        jif_foreground, optical_flows_mask, optical_flows, resx, number_of_frames, True, uv_foreground)
    uv_foreground_forward_should_match = model_F_mapping(xyt_foreground_forward_should_match.to(device))
    loss_flow_next = (uv_foreground_forward_should_match - uv_foreground_forward_relevant).norm(dim=1) * resx / ( 2 * uv_mapping_scale)

    # Backward flow:
    uv_foreground_backward_relevant, xyt_foreground_backward_should_match, relevant_batch_indices_backward = get_corresponding_flow_matches(
        jif_foreground, optical_flows_reverse_mask, optical_flows_reverse, resx, number_of_frames, False, uv_foreground)
    uv_foreground_backward_should_match = model_F_mapping(xyt_foreground_backward_should_match.to(device))
    loss_flow_prev = (uv_foreground_backward_should_match - uv_foreground_backward_relevant).norm(dim=1) * resx / (
                2 * uv_mapping_scale)

    if use_alpha:
        flow_loss = (loss_flow_prev * alpha[relevant_batch_indices_backward].squeeze()).mean() * 0.5 + (
            loss_flow_next * alpha[relevant_batch_indices_forward].squeeze()).mean() * 0.5
    else:
        flow_loss = (loss_flow_prev).mean() * 0.5 + (loss_flow_next).mean() * 0.5

    return flow_loss

'''
jif_foreground를 가상으로 만들어서 shape를 확인해보니 [3,82944,1]이다. 무슨 의미일까.
jif_all을 가상으로 만들어서 shape를 확인해보니 [3, 23131425]이다.
jif_current를 가상으로 만들어서 shape를 확인해보니 [3, 10000, 1]이다.

하나하나 확인해보니 다음과 같다는 결론을 내렸다.
첫 번째 3은 각각 resx, resy, frame_num을 가리킨다.
두번째 숫자는 한 batch에 포함되는 수를 나타낸다.
세번째 숫자는 의미없는 차원이다. (squeeze 해서 쓸 수 있다.)
'''
def get_corresponding_flow_matches(jif_foreground, optical_flows_mask, optical_flows, resx, number_of_frames,
                                   is_forward, uv_foreground, use_uv=True):
    batch_forward_mask = torch.where(
        optical_flows_mask[jif_foreground[1, :].squeeze(),
        jif_foreground[0, :].squeeze(), jif_foreground[2, :].squeeze(), :])
    forward_frames_amount = 2 ** batch_forward_mask[1]
    # 마스킹이 되어있는 batch indices를 나타낸다.
    relevant_batch_indices = batch_forward_mask[0]
    # 마스킹이 되어있는 batch index들에 해당하는 jif 점들을 나타낸다.
    jif_foreground_forward_relevant = jif_foreground[:, relevant_batch_indices, 0]
    forward_flows_for_loss = optical_flows[jif_foreground_forward_relevant[1], jif_foreground_forward_relevant[0], :,
                             jif_foreground_forward_relevant[2], batch_forward_mask[1]]

    if is_forward:
        jif_foreground_forward_should_match = torch.stack(
            (jif_foreground_forward_relevant[0] + forward_flows_for_loss[:, 0],
             jif_foreground_forward_relevant[1] + forward_flows_for_loss[:, 1],
             jif_foreground_forward_relevant[2] + forward_frames_amount))
    else:
        jif_foreground_forward_should_match = torch.stack(
            (jif_foreground_forward_relevant[0] + forward_flows_for_loss[:, 0],
             jif_foreground_forward_relevant[1] + forward_flows_for_loss[:, 1],
             jif_foreground_forward_relevant[2] - forward_frames_amount))

    xyt_foreground_forward_should_match = torch.stack((jif_foreground_forward_should_match[0] / (resx / 2) - 1,
                                                       jif_foreground_forward_should_match[1] / (resx / 2) - 1,
                                                       jif_foreground_forward_should_match[2] / (
                                                               number_of_frames / 2) - 1)).T

    if use_uv:
        uv_foreground_forward_relevant = uv_foreground[batch_forward_mask[0]]
        return uv_foreground_forward_relevant, xyt_foreground_forward_should_match, relevant_batch_indices
    else:
        return xyt_foreground_forward_should_match, relevant_batch_indices


def get_corresponding_flow_matches_all(jif_foreground, optical_flows_mask,
                                       optical_flows, resx, number_of_frames,
                                       use_uv=True):
    jif_foreground_forward_relevant = jif_foreground

    # optical_flows :      shape ([resy, resx, 2, number_of_frames, 1])
    forward_flows_for_loss = optical_flows[jif_foreground_forward_relevant[1], jif_foreground_forward_relevant[0], :,
                                           jif_foreground_forward_relevant[2], 0].squeeze()

    # optical_flows_mask:  shape ([resy, resx, number_of_frames, 1])
    forward_flows_for_loss_mask = optical_flows_mask[
        jif_foreground_forward_relevant[1], jif_foreground_forward_relevant[0],
        jif_foreground_forward_relevant[2], 0].squeeze()

    jif_foreground_forward_should_match = torch.stack(
        (jif_foreground_forward_relevant[0].squeeze() + forward_flows_for_loss[:, 0],
         jif_foreground_forward_relevant[1].squeeze() + forward_flows_for_loss[:, 1],
         jif_foreground_forward_relevant[2].squeeze() + 1))

    # xyt는 jif가 [-1, 1] 사이의 값을 가지도록 normalize 해준 것이다.
    xyt_foreground_forward_should_match = torch.stack((jif_foreground_forward_should_match[0] / (resx / 2) - 1,
                                                       jif_foreground_forward_should_match[1] / (resx / 2) - 1,
                                                       jif_foreground_forward_should_match[2] / (
                                                           number_of_frames / 2) - 1)).T

    if use_uv:
        return xyt_foreground_forward_should_match, forward_flows_for_loss_mask > 0
    else:
        return 0


'''
논문 Eq. 12에서 소개한 alpha optical flow loss를 계산한다.
'''
def get_optical_flow_alpha_loss(model_alpha,
                                jif_foreground, alpha, optical_flows_reverse,
                                optical_flows_reverse_mask, resx, number_of_frames,
                                optical_flows, optical_flows_mask, device):
    # Forward flow
    xyt_foreground_forward_should_match, relevant_batch_indices_forward = get_corresponding_flow_matches(
        jif_foreground, optical_flows_mask, optical_flows, resx, number_of_frames,True, 0, use_uv=False)
    alpha_foreground_forward_should_match = model_alpha(xyt_foreground_forward_should_match.to(device))
    alpha_foreground_forward_should_match = 0.5 * (alpha_foreground_forward_should_match + 1.0)
    alpha_foreground_forward_should_match = alpha_foreground_forward_should_match * 0.99
    alpha_foreground_forward_should_match = alpha_foreground_forward_should_match + 0.001
    loss_flow_alpha_next = (alpha[relevant_batch_indices_forward] - alpha_foreground_forward_should_match).abs().mean()

    # Backward loss
    xyt_foreground_backward_should_match, relevant_batch_indices_backward = get_corresponding_flow_matches(
        jif_foreground, optical_flows_reverse_mask, optical_flows_reverse, resx, number_of_frames, False, 0, use_uv=False)
    alpha_foreground_backward_should_match = model_alpha(xyt_foreground_backward_should_match.to(device))
    alpha_foreground_backward_should_match = 0.5 * (alpha_foreground_backward_should_match + 1.0)
    alpha_foreground_backward_should_match = alpha_foreground_backward_should_match * 0.99
    alpha_foreground_backward_should_match = alpha_foreground_backward_should_match + 0.001
    loss_flow_alpha_prev = (alpha_foreground_backward_should_match - alpha[relevant_batch_indices_backward]).abs().mean()

    return (loss_flow_alpha_next + loss_flow_alpha_prev) * 0.5

'''
시각화용 loss 계산
논문 Eq. 12에서 소개한 alpha optical flow loss를 모든 픽셀에 대하여 계산한다.
'''
def get_optical_flow_alpha_loss_all(model_alpha,
                                    jif_foreground, alpha, resx,
                                    number_of_frames, optical_flows, optical_flows_mask, device):
    xyt_foreground_forward_should_match, relevant_batch_indices_forward = get_corresponding_flow_matches_all(
        jif_foreground, optical_flows_mask, optical_flows, resx, number_of_frames)
    alpha_foreground_forward_should_match = model_alpha(xyt_foreground_forward_should_match.to(device))
    alpha_foreground_forward_should_match = 0.5 * (alpha_foreground_forward_should_match + 1.0)
    alpha_foreground_forward_should_match = alpha_foreground_forward_should_match * 0.99
    alpha_foreground_forward_should_match = alpha_foreground_forward_should_match + 0.001

    loss_flow_alpha_next = (alpha - alpha_foreground_forward_should_match).abs()
    loss_flow_alpha_next[relevant_batch_indices_forward == False] = 0

    return loss_flow_alpha_next




