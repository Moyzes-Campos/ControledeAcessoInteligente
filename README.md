# Controle de acesso inteligente

Aplicação para acompanhar pessoas na câmera, reconhecer os rostos cadastrados, registrar entradas e saídas e controlar a sinalização e o portão pelo ESP32. O uso diário é feito pelo `executar.ps1`.

A configuração atual usa **YOLO26** para detectar pessoas, **OC-SORT** para acompanhá-las, **SCRFD** para detectar rostos e **ArcFace** para reconhecer identidades.

## Preparar o ambiente

Use Windows x64, PowerShell, Python 3.10 e uma GPU NVIDIA com driver e CUDA compatíveis com as dependências de [requirements.txt](requirements.txt). Para compilar o InsightFace no Windows, podem ser necessárias as ferramentas C++ do Visual Studio Build Tools.

O script procura o Python e a engine YOLO na pasta **acima do projeto**. Mantenha esta estrutura:

```text
pasta_de_trabalho/
  .venv/
  models/yolo_v26/.engine/yolo26x_fp16.engine
  cargil_project_controledeacesso/
    executar.ps1
    requirements.txt
    faces/
    models/insightface_buffalo_l/
```

A engine YOLO externa precisa ser disponibilizada separadamente nesse caminho; ela não está incluída neste repositório. Os modelos faciais ficam dentro do projeto. Após clonar, baixe os arquivos gerenciados pelo Git LFS:

```powershell
git lfs install
git lfs pull
```

Na pasta do projeto, crie o ambiente se ele ainda não existir e instale as dependências:

```powershell
py -3.10 -m venv ..\.venv
..\.venv\Scripts\python.exe -m pip install --upgrade pip
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps onnxruntime-gpu==1.18.0 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

O último comando instala o ONNX Runtime para CUDA 12, compatível com o PyTorch deste projeto. Não é necessário ativar a `.venv` para usar o script.

As engines TensorRT dependem da GPU e da versão do TensorRT. Ao trocar de computador, providencie uma engine YOLO compatível. Para os modelos faciais, `-FaceRuntime auto` tenta usar as engines locais e, se forem incompatíveis, utiliza ONNX Runtime. As condições de uso dos pesos estão em [models/insightface_buffalo_l/README.md](models/insightface_buffalo_l/README.md).

## Executar

Abra o terminal na pasta do projeto. Para usar a câmera RTSP com a janela de acompanhamento e o ESP32:

```powershell
.\executar.ps1 -InputVideo "rtsp://USUARIO:SENHA@IP_CAMERA:554/cam/realmonitor?channel=1&subtype=0" -Show -EspUrl "http://192.168.3.1"
```

Substitua o usuário, a senha e o IP pelos dados da câmera. O computador precisa alcançar tanto a câmera quanto o ESP32 pela rede. `-InputVideo` é obrigatório; se executar apenas `.\executar.ps1`, o PowerShell solicitará a fonte.

Para usar a webcam ou testar um arquivo sem acionar o ESP32:

```powershell
.\executar.ps1 -InputVideo "0" -Show -NoEsp
.\executar.ps1 -InputVideo "C:\videos\teste.mp4" -Show -NoEsp -Database "data\teste.sqlite3"
```

O banco separado evita misturar os acessos de teste com os registros de uso. Encerre com **Q** ou **Esc** na janela, ou **Ctrl+C** no terminal. Sem `-Show`, não há janela. A gravação de vídeo fica desligada; adicione `-SaveVideo` para habilitá-la.

## Cadastrar pessoas

Coloque as fotos em `faces/NomeDaPessoa/`. O nome da pasta será a identidade exibida e registrada no banco:

```text
faces/
  Maria/
    frente.jpg
    lado.jpg
```

Use fotos nítidas, com um rosto por imagem e, preferencialmente, duas ou mais fotos por pessoa. São aceitos JPG, JPEG, PNG e BMP. Reinicie a aplicação após alterar o cadastro. O reconhecimento utiliza as fotos; o CSV da pasta não é necessário para carregar as identidades. Veja [faces/README.md](faces/README.md).

## Linha e registros de acesso

A entrada ou saída é registrada quando o **centro da caixa da pessoa cruza a linha**. Reconhecer um rosto sem cruzamento não abre uma visita. A entrada reconhecida abre a visita e a saída reconhecida a fecha pelo nome, mesmo que a pessoa tenha outro ID de rastreamento. Desaparecer da imagem não fecha a visita; visitas abertas permanecem após reiniciar o programa.

O script lê `outputs/linha/linha.json`. Sem esse arquivo, usa a linha padrão. Para outra posição, crie o arquivo com esta estrutura:

```json
{
  "line_normalized": [0.489844, 0.155556, 0.500391, 0.890972],
  "invert": false,
  "hysteresis_pixels": 18
}
```

As coordenadas `[x1, y1, x2, y2]` variam de 0 a 1 e correspondem à imagem após a rotação. A seta `ENTRADA` indica o sentido; use `-InvertLine` para invertê-lo. A pasta `outputs` é ignorada pelo Git, portanto essa configuração local precisa ser copiada separadamente para outro computador ou informada por `-LineConfig`.

## ESP32 e portão

A comunicação está habilitada por padrão em `http://192.168.3.1`. Use `-EspUrl` para outro endereço ou `-NoEsp` para executar sem comandos ao dispositivo.

| Situação | Sinalização | Portão |
| --- | --- | --- |
| Nenhuma pessoa visível | Laranja (GPIO5) | Bloqueado |
| Alguma pessoa desconhecida | Vermelho (GPIO6) | Bloqueado |
| Ao menos uma pessoa visível e todas reconhecidas | Verde (GPIO7) | Liberado |
| Desconhecido cruza a linha | Buzzer (GPIO4) por 3 segundos | Conforme a situação acima |

O relé do portão usa GPIO1: desligado bloqueia e ligado libera. O cliente exige firmware compatível com os comandos de GPIO e de estado do portão. Ao encerrar normalmente, solicita o bloqueio. Se a comunicação cair ou o processo for encerrado à força, o firmware atual pode manter o último estado do relé; o cliente não garante bloqueio nessa situação.

## Ajustes de uso

Adicione as opções ao mesmo comando de execução:

| Opção | Uso | Padrão |
| --- | --- | --- |
| `-Rotate -90` | Corrigir a orientação da imagem; aceita -90, 0, 90 ou 180 | 0 |
| `-MaxFps 20` | Limitar o processamento ao vivo; 0 acompanha o FPS da câmera | 0 |
| `-FaceThreshold 0.40` | Limiar de reconhecimento ArcFace; valores maiores exigem maior similaridade | 0.40 |
| `-FaceRotations 0` | Buscar rostos só na orientação original | 0, 90, 180, -90 |
| `-PreviewMaxSide 1280` | Limitar o tamanho da prévia | 1280 |
| `-AlarmSeconds 5` | Duração do buzzer em segundos | 3 |
| `-LineConfig "config\linha.json"` | Usar outra configuração da linha | `outputs\linha\linha.json` |
| `-Database "data\teste.sqlite3"` | Escolher o banco de acessos | `data\acessos.sqlite3` |
| `-Output "outputs\teste"` | Escolher a pasta de resultados | `outputs\execucao` |
| `-SaveVideo` | Gravar o vídeo anotado | Desligado |

## Arquivos gerados

O banco `data/acessos.sqlite3` guarda as visitas (entrada, saída e permanência) na tabela `acessos` e todos os cruzamentos, inclusive desconhecidos, na tabela `cruzamentos`.

Em `outputs/execucao/`, ou na pasta escolhida por `-Output`, são gravados:

- `eventos.jsonl`: eventos por quadro, identidades e cruzamentos.
- `resumo.json`: contagens e informações da execução.
- `resultado.mp4`: vídeo anotado, somente com `-SaveVideo`.

Use uma pasta de saída diferente para cada execução cujo resultado queira preservar.
