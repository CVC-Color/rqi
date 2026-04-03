from rqi import RQI

model = RQI(pretrained=True)

# score = model(test_image, gt_image)
score = model("imgs/0801_BSRGAN.png", "imgs/0801_GT.png")
print(score)