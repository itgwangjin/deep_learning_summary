# Deep learning model for SFDI

@(첫 번째 노트북)[Iterative, Nearest search, Deep learning, fminsearch, pdist2, Cramer-Rao lower bound, Adam optimization, Levenberg Marquardt optimization, Monte-Carlo]
**논문 명** : Deep learning model for ultrafast multifrequency optical property extractions for spatial frequency domain imaging

**자료 링크** :  [PDF](https://www.researchgate.net/publication/328966209_Deep_learning_model_for_ultrafast_multifrequency_optical_property_extractions_for_spatial_frequency_domain_imaging) 

**목차**

[TOC]

## 0. Introduce
**Spatial frequency domain imaging (SFDI)** 는 label-free라는 특성 넓은 tissue optical property maps를 제공하는 것 때문에 biomedical imaging의 주요방식이다.
대부분의 SFDI는 광학특성을 추출하기 위해서 2개의 공간 주파수(spatial frequencies)(=$2 − f_x$ )를 사용했었다. 

2개 이상의 $multi-f_x$를 사용할 경우
- **장점**
	- 정확성(Accuracy)을 크게 향상.
	- 광학특성 추정하는데 불확실성(uncertainty) 절감.

- **단점**
	- Inversion algorithm사용 시 낮은 속도.
- **Research solution**
	- Deep learning을 통해 Speed 문제를 극복하여 이전에 사용한 방법보다 300x ~ 100,000x 빠르게 만들었다.

---
##  1. SFDI
### (1) What is SFDI
생물학적 세포(tissues)는 다양한 발색단(chromophores)으로 구성되어있습니다.
발색단은 특성화된 wavelengths를 흡수하는 분자구조로 되어있는데요
만약 당신이 세포의 흡수하는 빛을 측정한다면 당신은 발색단(chrompophores)의 농도를 측정할수 있고 이것
이 발색단은 매우 유용한 application입니다 왜냐하면 세포안의 dominant한 발색단 중 일부는 중요한 임상 매개변수의 지표이기 때문입니다.(ex. oxygenation)

그러나 이것은 turbid하기 때문에 absorption을 정량화 하는 것은 어렵습니다.
tissue에서의 빛의 감쇠(attenuation)는 absorption과 scattering이라는 특성의 기능을 가지고 있습니다. 
SFDI는 absorption과 scattering의 효과를 분리하여 발색단(chrompohores)을 정량화하는 기술입니다.
이 기술은 다른 패텅늬 빛을 세포(tissue)에 쏘고, 방출된 빛을 촬영하고 획득한 영상을 분석합니다.
Frangioni Lab에서는 성형수술에 사용했습니다.특히 MIT의 경우 SFDI기술을 이용하여 유방 재건 수술 이전에
피부 플랩 이식의 생존능력을 조기에 평가할수있는 SFDI능력을 테스트했습니다.

1) Optical property
광학 특성에는 2종류가 있다 **흡수**(Absorption($μ_a$))과 **산란**(scattering($μ`_s$))
이것은 tissue의 기능과 분자(molecular)구성에 중요한 정보를 주는데 병을 진단하는데 광범위하게 쓰인다.\


### (2) 사용원리

![@The dataflow of SFDI optical property extraction| center | 600x0 ](./fi1.PNG)
1. SFDI는 변조(modulated)된 조명 패턴을 tissue에 쏘고 그에 반사되는 images를 카메라로 담는다.
2. 반사된 이미지를 복조(demodulate)하고 정량을 정한다(calibrate).
3. 해당 이미지를 각 공간주파수(spatial frequency)에 퍼뜨린다.
4. 이거를 $R_d(f_x)$라고 명명한다.
5. $R_d$ vector를 inverse model의 input으로 집어넣는다.
(*invers model은각 조명 파장에 대해  $μ_a$와 $μ`_s$를 픽셀단위로 추출하는 데 사용된다. )
### (3) 요구사항
Inversion을 위해서는 최소 2개 공간 주파수가 필요합니다.
대부분 AC $f_x$와 쌍을 이루는 DC($0mm^{-1}$)를 사용했는데
우리의 이전연구에서 $f_x$의 선택이 OPs를 추출하는데 정확성과 불확실성에 큰 영향을 미친다는 것을 알아냈고 $1-f_x$는 광범위한 tissue type에 적합하지 않았다는 것을 발견했다.
반면 2개 이상의 $f_x$에서 **특정 상황**에서는 OP 추출에서의 error가 최소화됨을 발견했고 아래 사진에서 알 수 있듯이
<p style="text-align: center;">
    <img src="./1553605044026.png" width="1000%"/>
</p>
 $2-f_x$ ([0, 0.1] $mm^{-1}$)와 $5-f_x$ ([0, 0.05, 0.1, 0.2, 0.4]$mm^{-1}$)를 사용했을 때 $5-f_x$에서의 성능이 $2-f_x$ 보다 월등히 좋음을 알 수 있다.
> OP uncertainties는 **Cramer-Rao lower bound**를 사용했다.

생물학적 조직은 630nm ~ 950nm 사이에서 학문적으로 광학특성 값(Iterature OP values)을 대표함을 컬러마커를 통해 알 수 있다.
표에서 알 수 있다시피  $2-f_x$를 사용할 때, 몇몇 tissue type(ex. skin) 에서 측정 불확실성이 15%나 높다는 것을 알 수 있다.
이 불확실성은 $5-f_x$를 사용했을 때 55% 이상 줄일 수 있다.
OP uncertainties에서의 잠재적 향상에도 불구하고 사용 못하는 이유가 있다.
$multi-f_x$는 inversion step에서 시간 소모가 크기 때문에 실시간 SFDI는 $2-f_x$만 사용하도록 남아있었다.
현존하는 $multi-f_x$ SFDI inverse model은 일반적으로 **Iterative search algorithm**과 **Nearest search algorithm**에서 사용한다.

----------

## 2. Algorithm
### (1) Iterative search algorithm
#### 1) 원리
이 방법은 Matlab에서 **fminsearch** 함수를 사용했다.
![Alt text](./a.png)

1. 초기 예상되는 OPs`를 만든다
2. **Monte-Carlo** 기반 모델을 사용하여 일치하는 $R_d$를 계산한다
3. 계산된 $R_d$는 실제 측정된 $R_d$와 비교해본다.
4. Convergence에 도달할 때까지 Iterative algorithm을 사용하여 차이를 최소화시킨다.

#### 2) 특징
장점 : 
단점 : 반복특성 때문에 시간 소모가 극심해진다.
image의 각 pixel을 반복해야 한다.  

### (2) Nearest search algorithm
#### 1) 원리
이 방법은 Matlab에서 **pdist2** 함수를 사용했으며 Euclidean distance로 쌍을 계산했다.
![Alt text](./b.png)

1. 많은 OP 조합을 forward model에 넣는다. (미리 $R_d$ set이 일치하는지 계산하기 위해서)
2. 측정된 $R_d$값은  위에서 미리 계산된 $R_d$값과 비교해본다.
3. 유클리드로 가장 오차가 적은 것을 결정한다.
#### 2) 특징
* 장점 
	* Iterative시에 사용한 방법보다 속도가 빠르다.
	*  $R_d$를 한번 수행하기 때문에 매우 적은 LUT(lookup table)를 요구한다.
	> 원래알고리즘을 효율적으로 쓰기 위해서 원래 LUT에서의 각 차원에서 **직접 좌표조회**(The direct coordinate lookup)방법을 수행하는데,   공간 주파수가 늘어날 수록 기하급수적으로 늘어난다.
	(ex, $5-f_x$의 500 $R_d$가 있는 LUT의 값은 각 $f_x$는 최대 $3*10^{13}$ 를 가진다.)
* 단점 
	* 정확성이 미리 계산한 OP sampling의 밀도에 영향을 받는다.
. . . . . . . => 그럼 OP sampling 밀도를 높이면 되지 않을까?
. . . . . . . => 밀도를 높이게 되면 memory 사용량 증가, 계산 속도가 느려진다.
> Nearest search algorithm is distinct from the direct coordinate lookup method proposed for 2 − fx by Angelo et al [[PDF]](https://www.spiedigitallibrary.org/journals/Journal-of-Biomedical-Optics/volume-21/issue-11/110501/Ultrafast-optical-property-map-generation-using-lookup-tables/10.1117/1.JBO.21.11.110501.full) 
### (3) Deep Neural Network(DNN)
#### 1) 특징
**Deep learning**은 고차원공간에서 자동으로 비선형 패턴을 탐지하고 근사화시킨다.
![Alt text](./c.png)
* 과거 문제점 :   Deep learning 적용한 SFDI는 hidden layer를 1개만 사용하는 structure를 사용했다. 
 => estimation error가 많이 났다. (관련 논문 : [Spatial frequency domain spectroscopy of
two layer media](https://www.spiedigitallibrary.org/journals/Journal-of-Biomedical-Optics/volume-16/issue-10/107005/Spatial-frequency-domain-spectroscopy-of-two-layer-media/10.1117/1.3640814.full))
* new field : optical parameter estimation
* Idea : $multi - f_x $ SFDI inversion에 DNN사용해 n차원 $R_d$ space(n=$f_x$ 수)를 2D OP space로 바꾸는 것이다.
#### 2) 구성
* 6 fully connected hidden layers
* 10 neurons in each layer
* $\tanh$ sigmoid transfer function
( ($-\infty, +\infty$) data -> ($-1, 1$) data )
* Input : $R_d$ values at different $f_x$
* Output : $\mu_a$ and $\mu'_s$
* training data : _a semi-infinite homogeneous "white" MC model _  [[PDF]](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-19-20-19627)
$\mu_a$ : 0.001 ~ 0.25 $mm^{-1}$ 0.001씩 증가
$\mu'_s$ : 0.01 ~ 3 $mm^{-1}$ 0.01씩 증가
>these training data were also used as the precomputed dataset for the nearest-search method
* Hyperparameters : tuned in Keras using Adam optimization [[PDF]](https://arxiv.org/pdf/1412.6980.pdf)
<!--옵티마이저에 대해서 좀 더 자세하게 다루는게 좋을것 같다. -->
* learning rate : 0.001
* batch size : 128
* minmization of loss function : mean squared error
* To reduce overfitting : Levenberg–Marquardt optimization with Bayesian regularization
* completed training epochs : 2000
* time : 1h
* overfitting test : overfitting이란 training set에서는 매우 잘 작동하는데 test set에서는 잘 작동하지 않는것인데
 여기 논문에서는 10,000 OP combinations 중
$μ_a$는 [0, 0.25] mm$^{-1} $
$μ'_s$는 [0, 3.00] mm$^{-1}$
범위에서 를 랜덤으로 선택하였고 corresponding $5-f_x$ decision : Monte Carlo forward model방식을 사용하여
![Alt text| center | 500x0](./삭용.PNG)
blue dot : randomly generated Ops
red lines : linear fit
overfitting이 발생하지 않았다고 할 수있다.

![Alt text| center | 450x0](./삭제.PNG)
expected OPs and estimated OPs 
* accuracy : DNN은 iterative나 nearest-search method보다 더 좋은 정확성을 띈다.
iterative가 OP space 전체를 search해야하는것과는 달리 DNN은 일부 sampled OP set만 씀에도 같은 성능을 자랑한다.
* Comparison of speed
![Alt text](./삭제용.PNG)
100x100 $R_d$ maps (10,000 inversions from $R_d$ to OPs)
위로써 nearest search보다 300배 빠르고 iterative보다 $10^5$배 빠르다.

----------

## 3. TEST
### (1) Phantom measurements
* SFDI measurements from five spatial frequencies (0, 0.05, 0.1, 0.2, and 0.4 mm−1) were acquired at two wavelengths (659 and 851 nm)
* DNN으로 인해서 Acquisition, demodulation, calibration, optical property inversion, chromophore extraction for oxy- and deoxy- hemoglobin, and visualizations를 실시간으로 가능하게 됐다.
* The image size was 520 × 696, and the exposure time was 16 ms for 659 nm and 36 ms for 851 nm.
* 
### (2) Cuff occlusion
* The cuff was applied to the upper arm, and the inner arm was measured with SFDI
* 측정은 5s회씩 총 8분 진행되었다. -> 480/5 = 135s
* the final frame of the recorded video and the time series of hemoglobin changes calculated from the region-of-interest (red-dashed boxes)



----------
