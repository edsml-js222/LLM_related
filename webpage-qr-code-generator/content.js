(function() {
  console.log('QR Code Generator script is running');

  // 创建容器div
  const container = document.createElement('div');
  container.style.position = 'fixed';
  container.style.right = '20px';
  container.style.bottom = '20px';
  container.style.zIndex = '9999999';  // 增加 z-index 确保在最上层
  container.style.backgroundColor = 'white';  // 添加白色背景
  container.style.padding = '10px';  // 添加一些内边距
  container.style.border = '1px solid black';  // 添加边框以便于识别

  // 创建QR码div
  const qrDiv = document.createElement('div');
  qrDiv.id = 'page-qr-code';
  container.appendChild(qrDiv);

  // 将容器添加到页面
  document.body.appendChild(container);

  // 检查 QRCode 是否可用
  if (typeof QRCode === 'undefined') {
    console.error('QRCode library is not loaded');
    qrDiv.textContent = 'QR Code library failed to load';
  } else {
    console.log('Generating QR code');
    // 生成QR码
    new QRCode(qrDiv, {
      text: window.location.href,
      width: 256,
      height: 256,
      colorDark: "#000000",
      colorLight: "#ffffff",
      correctLevel: QRCode.CorrectLevel.H
    });
  }

  console.log('QR Code Generator script finished');
})();
