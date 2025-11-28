# Pro-Rectified-Flow-Match-

Fast high-fidelity MRI super-resolution using Rectified Flow.
Final project for EEL6935

## Highlights
**630× faster** than standard diffusion models

**31× faster** than DDIM samplers

**39.01 dB PSNR** on 11,475 test samples

Evaluated on 1,151 clinical patients



## Technical Features
- Distributed training with Ray
- Patient-level stratification (no data leakage)
- EMA weight averaging
- Comprehensive evaluation (PSNR, SSIM, LPIPS, FID)

## Results
<img src="assets/comparisonFigure.png" width="800"/>

*Figure: Comparison of interpolation methods. From left to right: Ground Truth, Bilinear, DDIM, Ours (RF-Euler-3). Our method recovers fine anatomical details with minimal artifacts.*



## Quantitative Results

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>Steps</th>
      <th>PSNR ↑</th>
      <th>SSIM ↑</th>
      <th>LPIPS ↓</th>
      <th>FID ↓</th>
      <th>Time (s) ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Bilinear Interpolation</td>
      <td>-</td>
      <td>36.58</td>
      <td><u>0.888</u></td>
      <td>0.110</td>
      <td>15.61</td>
      <td><b>&lt;0.01</b></td>
    </tr>
    <tr>
      <td>DDPM</td>
      <td>1000</td>
      <td>30.95</td>
      <td>0.832</td>
      <td>0.117</td>
      <td>23.15</td>
      <td>2.46</td>
    </tr>
    <tr>
      <td>DDIM</td>
      <td>50</td>
      <td>36.74</td>
      <td><b>0.890</b></td>
      <td>0.089</td>
      <td><u>3.16</u></td>
      <td>0.12</td>
    </tr>
    <tr style="background-color: #f0f8ff;">
      <td><b>Ours (RF-Euler)</b></td>
      <td><b>3</b></td>
      <td><b>39.01</b></td>
      <td>0.868</td>
      <td><b>0.067</b></td>
      <td>6.14</td>
      <td><b>&lt;0.01</b></td>
    </tr>
    <tr style="background-color: #f0f8ff;">
      <td><b>Ours (RF-Euler)</b></td>
      <td><b>20</b></td>
      <td><u>37.48</u></td>
      <td>0.824</td>
      <td><u>0.073</u></td>
      <td><b>2.34</b></td>
      <td><u>0.08</u></td>
    </tr>
  </tbody>
</table>

*Quantitative comparison on 11,475 test images from 201 patients. **Bold** = best result, <u>underline</u> = second best. Our method achieves highest PSNR (39.01 dB) and lowest LPIPS (0.067) while maintaining real-time inference speed.*
