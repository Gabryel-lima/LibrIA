# Unica forma temporária sem wecam que encontrei foi utilizando o droidcam legacy

**Pré-requisitos**  
Certifique‑se de que o seu Ubuntu é 64 bits e que você tem os seguintes pacotes instalados:  
```bash
sudo apt update
sudo apt install gcc make adb usbmuxd linux-headers-$(uname -r)
```  
Esses pacotes fornecem o compilador, as ferramentas de construção, o ADB (para conexão USB) e os headers do kernel necessários para o módulo de vídeo ([Diolinux](https://diolinux.com.br/aplicativos/droidcam-webcam-com-o-smartphone.html?utm_source=chatgpt.com)).

---

## 1. Download e instalação do cliente e módulos  
1. Abra um terminal e vá para um diretório temporário:  
   ```bash
   cd /tmp/
   ```  
2. Baixe o pacote mais recente:  
   ```bash
   wget https://files.dev47apps.net/linux/droidcam_latest.zip
   ```  
3. Extraia e entre na pasta:  
   ```bash
   unzip droidcam_latest.zip -d droidcam && cd droidcam
   ```  
4. Execute os scripts de instalação em sequência:  
   ```bash
   sudo ./install-client    # instala o aplicativo PC
   sudo ./install-video     # compila e carrega o módulo v4l2loopback-dc
   sudo ./install-sound     # (opcional) configura áudio via ALSA Loopback
   ```  
Esses scripts instalam o cliente, o módulo de vídeo adaptado (v4l2loopback‑dc) e, opcionalmente, o suporte de som ([Diolinux](https://diolinux.com.br/aplicativos/droidcam-webcam-com-o-smartphone.html?utm_source=chatgpt.com), [dev47apps.com](https://www.dev47apps.com/droidcam/linux/?utm_source=chatgpt.com)).

---

## 2. Executando o DroidCam no Ubuntu  
- **No smartphone**: instale o app DroidCam pela Play Store (Android) ou App Store (iOS).  
- **No desktop**:  
  - Abra o menu do GNOME (ou Dash) e pesquise por **DroidCam**, ou execute no terminal:
    ```bash
    droidcam
    ```  
  - No cliente, escolha **WiFi/LAN**, insira o IP e a porta exibidos no app do celular e clique em **Connect**.  
A partir daí, “DroidCam” aparecerá como dispositivo de vídeo em apps como Zoom, OBS e Skype ([Diolinux](https://diolinux.com.br/aplicativos/droidcam-webcam-com-o-smartphone.html?utm_source=chatgpt.com)).

---

## 3. Resolução personalizada  
O padrão é 640×480. Para alterar, recarregue o módulo com novos parâmetros (exemplo para 1280×720):  
```bash
sudo rmmod v4l2loopback_dc
sudo insmod /lib/modules/$(uname -r)/kernel/drivers/media/video/v4l2loopback-dc.ko width=1280 height=720
```  
Para tornar a configuração persistente, edite o arquivo `/etc/modprobe.d/droidcam.conf`, adicionando:  
```
options v4l2loopback_dc width=1280 height=720
```  
Assim, após cada reboot, o módulo carregará com a resolução desejada ([Diolinux](https://diolinux.com.br/aplicativos/droidcam-webcam-com-o-smartphone.html?utm_source=chatgpt.com), [dev47apps.com](https://www.dev47apps.com/droidcam/linux/?utm_source=chatgpt.com)).

---

## 4. Desinstalação  
Caso queira remover completamente o DroidCam do seu sistema:  
```bash
sudo /opt/droidcam-uninstall
```  
Isso apagará o cliente, o módulo e as configurações associadas ([Diolinux](https://diolinux.com.br/aplicativos/droidcam-webcam-com-o-smartphone.html?utm_source=chatgpt.com)).

---

> **Observação sobre áudio**  
> Embora exista o script `install-sound`, muitos usuários enfrentam instabilidades mantendo o DroidCam apenas para vídeo e usando um microfone dedicado para áudio. Se optar por utilizar o áudio do smartphone, esteja preparado para ajustar manualmente módulos ALSA ou PulseAudio/PipeWire ([dev47apps.com](https://www.dev47apps.com/droidcam/linux/?utm_source=chatgpt.com)).

Lembrando, inicie o droidcam em um terminal anterior, antes de passar o código.