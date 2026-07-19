# PINN 深海温度–压力拟合简报
抽样数据量：30000
最终总损失：6.5885e-01
最终数据损失：6.5885e-01
最终物理损失：8.3484e-08

设置：
{
  "use_fourier": true,
  "prior_coef": null,
  "epochs": 2000,
  "lr": 0.001,
  "lambda_phys_init": 0.1,
  "adapt_every": 200,
  "device": "cpu"
}

图表：figures/pinn_tp_fit.png, figures/pinn_residual_heatmap.png, figures/pinn_losses.png, figures/pinn_lambda_history.png
