# Controle de acesso por visão computacional

Pipeline offline/tempo real que combina:

- detecção de pessoas com YOLO11 (pose também é aceita);
- tracking persistente por OC-SORT (IoU + movimento + direção);
- detecção facial SCRFD (ou YuNet) e reconhecimento facial SFace;
- contagem por linha, visitas no SQLite e sinalização pelo ESP32;
- vídeo anotado, eventos por quadro em JSONL e resumo da execução.

## Ambiente

O projeto reutiliza a `.venv` da pasta raiz:

```text
01_Computer_Vision_Projects/
  .venv/
  cargil_project_controledeacesso/
```

Os scripts PowerShell já selecionam esse Python. Não é necessário ativar a
`.venv` manualmente.

## Cadastro de pessoas

Adicione fotos em `faces/NomeDaPessoa/`. Use duas ou mais fotos nítidas por
pessoa, preferencialmente com pequenas variações de ângulo. Veja
`faces/README.md`.

## Executar o DAV informado

```powershell
cd cargil_project_controledeacesso
.\executar_video_exemplo.ps1
```

Esse script aplica `-Rotate -90`, pois a câmera do DAV informado está montada
na orientação vertical.

Para exibir a janela durante o processamento:

```powershell
.\executar.ps1 -InputVideo "C:\caminho\camera.dav" -Show
```

A janela de visualizaÃ§Ã£o Ã© ajustada automaticamente Ã  Ã¡rea Ãºtil do monitor,
mantendo a proporÃ§Ã£o original do vÃ­deo. Ela tambÃ©m pode ser redimensionada
manualmente.

Webcam e RTSP também são aceitos:

```powershell
.\executar.ps1 -InputVideo "0" -Show
.\executar.ps1 -InputVideo "rtsp://admin:dnxa36912@192.168.3.71:554/cam/realmonitor?channel=1&subtype=0" -Show
```

## Saídas

Cada execução gera em `outputs/<nome>/`:

- `resultado.mp4`: gerado somente com `-SaveVideo` (ou `--save-video` no Python);
- `eventos.jsonl`: caixas, keypoints, identidade e confiança por quadro;
- `resumo.json`: resolução, FPS, quantidade de IDs e caminhos das saídas.

## Ajustes úteis

O CLI expõe `--pose-conf`, `--face-threshold`, `--face-interval`,
`--tracker-iou`, `--tracker-max-age`, `--imgsz`, `--device` e `--max-frames`.
Use `--start-frame` para iniciar a análise em um ponto específico do arquivo.
Use `--rotate -90`, `90` ou `180` quando a câmera estiver fisicamente girada.
Execute `..\.venv\Scripts\python.exe main.py --help` para ver tudo.

O limiar SFace padrão (`0.363`) é um ponto inicial. Calibre-o com imagens reais
da câmera antes de tomar decisões de liberação de acesso. A decisão física de
abrir porta/catraca não faz parte deste protótipo.

## Testes

```powershell
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Stream RTSP e grava??o opcional

A grava??o de v?deo fica desativada por padr?o para todas as fontes. Eventos e resumo continuam sendo salvos. Para gravar, adicione `-SaveVideo` ao `executar.ps1`.

RTSP usa FFmpeg e TCP por padr?o (respeitando `OPENCV_FFMPEG_CAPTURE_OPTIONS` quando j? definido). Como no projeto PPE, uma thread captura continuamente e a infer?ncia recebe apenas o quadro mais recente, evitando acumular atraso. Arquivos s?o lidos sequencialmente, sem descartar quadros. Em falhas de leitura, a c?mera ? reconectada automaticamente at? encerrar com Q/Esc na janela ou Ctrl+C no terminal. O tempo dos eventos ao vivo usa o tempo decorrido, pois quadros podem ser descartados.

Erros HEVC de refer?ncia podem persistir se a c?mera enviar quadros corrompidos. Nesse caso, experimente configurar H.264 na c?mera ou usar o substream (`subtype=1`) com menor resolu??o/bitrate.

O modelo padr?o agora ? `../models/yolo_v11/.engine/yolo11l_fp16.engine` (detec??o, sem keypoints). Use `--model` para escolher outro modelo; `--pose-model` continua aceito. Faces detectadas pelo YuNet t?m caixas verdes, inclusive desconhecidas. Quando n?o h? tracks, YuNet busca faces no quadro completo.

Na grava??o ao vivo, o MP4 mant?m a dura??o real desde a primeira imagem anotada, repetindo a ?ltima imagem exibida at? a pr?xima atualiza??o. Arquivos de entrada mant?m seu FPS original. A grava??o continua opcional com `-SaveVideo`.

A busca YuNet utiliza a caixa inteira da pessoa com margem, pois o rosto pode estar fora da regi?o superior em c?meras giradas. A leitura das fotos de cadastro suporta caminhos com acentos no Windows.

## Teste isolado do YuNet

Para testar somente a detecção facial no quadro inteiro, sem carregar YOLO,
tracking ou SFace:

```powershell
.\testar_yunet.ps1 -InputVideo "rtsp://usuario:senha@camera/stream" -Show
```

O teste desenha a caixa, os cinco landmarks e a confiança do YuNet. O limiar
padrão é `0.60`; compare com `-DetectScore 0.80` para reproduzir a configuração
do pipeline principal. Use `-Rotate 90` ou `-Rotate -90` para avaliar a imagem já
na orientação correta. `-SaveVideo` grava `outputs/teste_yunet/resultado_yunet.mp4`.
O resumo informa a porcentagem de quadros nos quais alguma face foi detectada.

## Detector facial SCRFD

O pipeline principal usa por padrão o SCRFD-10GF (`det_10g.onnx`) do pacote
InsightFace Buffalo_L. Para lidar com a câmera superior, cada região de pessoa é
avaliada em 0°, 90°, 180° e -90°; resultados repetidos são consolidados antes do
alinhamento e reconhecimento SFace. O limiar padrão é `0.50`.

```powershell
.\executar.ps1 -InputVideo "URL_RTSP" -Show
```

Para comparar com o detector anterior:

```powershell
.\executar.ps1 -InputVideo "URL_RTSP" -Show -FaceDetector yunet -FaceDetectScore 0.60
```

Os pesos oficiais pré-treinados do InsightFace são restritos a pesquisa e testes
não comerciais. Veja `models/insightface_buffalo_l/README.md`. Obtenha uma
licença apropriada antes de qualquer uso comercial ou em produção.

Neste ambiente com PyTorch CUDA 12.1/cuDNN 8, o ONNX Runtime compatível deve ser
instalado pelo repositório CUDA 12 oficial da Microsoft:

```powershell
python -m pip install onnxruntime-gpu==1.18.0 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

O reconhecimento facial é interrompido para um `track_id` assim que o SFace
confirma uma identidade. Tracks desconhecidos continuam sendo verificados no
intervalo configurado por `--face-interval`. Quando o track desaparece, sua
memória é removida; uma nova entrada recebe outro ID e volta a ser verificada.
Use `-FaceRecheckInterval N` para reverificar identidades conhecidas após `N`
quadros. O valor padrão `0` mantém a reverificação desativada.

## Desempenho do reconhecimento facial

Em RTSP e webcam, a detecção e o reconhecimento facial agora usam uma thread
separada. YOLO, tracking, contagem e visualização continuam atualizando enquanto
a tarefa facial trabalha. Há uma única tarefa em andamento, sem fila de quadros
antigos. Cada tarefa verifica uma pessoa por padrão, priorizando quem ainda não
foi verificado ou ficou mais tempo sem uma tentativa.

Pessoas desconhecidas são tentadas novamente com um intervalo mínimo de
`0.5` segundo por ID e respeitando `-FaceInterval` (padrão: 3 quadros).
Pessoas reconhecidas continuam dispensadas de novas tentativas, salvo quando
`-FaceRecheckInterval` é configurado. O resultado é associado às caixas do quadro
original e só permanece na memória enquanto o ID de tracking estiver ativo.

```powershell
.\executar.ps1 -InputVideo "URL_RTSP" -Show
.\executar.ps1 -InputVideo "URL_RTSP" -Show -FaceRetrySeconds 0.8 -FaceInterval 5
```

O SCRFD mantém as quatro rotações por padrão. Quando os rostos estiverem sempre
na orientação correta, compare `-FaceRotations 0`; isso reduz de quatro para uma
inferência por pessoa, mas pode perder rostos inclinados ou em outras orientações.
Use `-FaceRotations 0,90,180,-90` para a busca completa.

O ONNX Runtime facial limita o processamento interno a duas threads de CPU e desativa a espera
ativa dessas threads para reduzir a disputa com o restante do pipeline.
`resumo.json` registra os providers ativos, a quantidade de tarefas e os tempos
médios de detecção, extração da característica facial e comparação com o cadastro
em `face_processing`. `eventos.jsonl` registra tempos por quadro em `timings_ms`.

Arquivos de vídeo mantêm a análise facial síncrona para preservar a sequência
completa. Para comparar o comportamento anterior na câmera, use `-FaceSync`.
No modo assíncrono, a identificação chega após a conclusão da tarefa: durante
esse período a pessoa segue desconhecida e um cruzamento pode acionar o alerta.
O sistema já vincula a identificação posterior ao horário original do cruzamento
se o mesmo tracking continuar ativo. GPU e CPU ainda são compartilhadas; o FPS
final precisa ser medido na câmera e no computador usados.

Na validação local com YOLO TensorRT e SCRFD CUDA, uma imagem de 2560x1440
contendo uma pessoa foi repetida a 20 FPS, com 80 quadros por modo e a pessoa
mantida desconhecida para exercitar as tentativas. A média passou de 9.02 FPS
no modo síncrono para 20.17 FPS no assíncrono. O percentil 95 do tempo de
processamento por quadro passou de 237.45 ms para 34.40 ms. A comparação usa
as mesmas quatro rotações e não representa uma medição da câmera ao vivo.
Veja `outputs/diagnostico_fps_20261007/comparacao.json`.

## Engine TensorRT para o detector facial SCRFD

O `det_10g.onnx` pode ser executado como `det_10g_fp16.engine`. O modo padrão
`-FaceRuntime auto` usa a engine local quando ela estiver presente e compatível;
se não houver engine, TensorRT instalado ou compatibilidade com o ambiente,
o pipeline informa o motivo e segue com ONNX Runtime/CUDA.

```powershell
.\executar.ps1 -InputVideo "URL_RTSP" -Show -FaceRuntime tensorrt -MaxFps 20
.\executar.ps1 -InputVideo "URL_RTSP" -Show -FaceRuntime cuda -MaxFps 20
```

O modo `tensorrt` exige uma engine válida e informa erro se ela não puder ser
usada. `cuda` permite comparar com o detector ONNX anterior. `-FaceEngine`
aceita outro caminho de engine. A engine otimiza o SCRFD; o reconhecimento
SFace continua no OpenCV. As quatro rotações, os landmarks, o alinhamento e
os limiares são mantidos.

Para gerar ou recriar a engine no computador de destino:

```powershell
..\.venv\Scripts\python.exe exportar_scrfd_engine.py
..\.venv\Scripts\python.exe exportar_scrfd_engine.py --force
```

A exportação exige TensorRT 10 com bindings Python e CUDA, usa entrada
`1x3x640x640`, FP16 e até 1 GiB de workspace de construção por padrão.
O arquivo `.engine.json` acompanha a engine com hashes do ONNX e da engine,
ordem das saídas, versão do TensorRT e dados da GPU. Recrie a engine ao mudar
de computador ou versão do TensorRT. O runtime usa buffers CUDA e um stream
próprios para a tarefa facial, separados do contexto de execução do YOLO.

Na validação com RTX 3050 6GB e TensorRT 10.14.1, o quadro local da câmera
teve mediana de detecção reduzida de 155.80 ms (ONNX/CUDA) para 109.14 ms
(TensorRT FP16), considerando as quatro rotações. A etapa completa de
detecção e reconhecimento passou de 198.08 ms para 149.59 ms. Foram usadas
três execuções de aquecimento e quinze medições por modo. Nas duas imagens
com rostos testadas, as identidades foram mantidas, e a maior diferença nas
coordenadas de caixas e landmarks foi de 0.35 pixel. Isso não substitui a
validação com pessoas, iluminação e poses reais da câmera.
Os resultados estão em `outputs/diagnostico_scrfd_engine_20261007/comparacao.json`.

Referências: [API Python do TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html)
e [compatibilidade do TensorRT no ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html).

## FPS da câmera e limite de processamento

Sem pessoas detectadas, o pipeline não inicia novas tarefas faciais. O YOLO
continua procurando pessoas, junto da captura e das atualizações de estado.
A captura ao vivo aguarda uma nova sequência de quadro antes de entregá-lo;
uma imagem já consumida não é processada repetidamente.

O indicador `FPS` mostra a frequência real de atualização, medida em uma janela
de um segundo e incluindo a espera por novos quadros. O campo `Processamento`
mostra separadamente o tempo da análise em milissegundos. Uma análise de 20 ms
tem capacidade de 50 quadros por segundo, mas uma câmera de 20 FPS entrega apenas
20 imagens por segundo.

O processamento ao vivo passa a respeitar o FPS informado pela câmera por padrão.
Use `-MaxFps 20` para estabelecer um teto explícito ou `-MaxFps 10` para reduzir
as inferências do YOLO em uma câmera de 20 FPS. O limite efetivo é o menor entre
o valor configurado e o FPS válido informado pela câmera. `-MaxFps 0` seleciona
o modo automático; se a câmera não informar um FPS válido, a captura acompanha
a chegada de quadros e só aplica um teto quando configurado explicitamente.

```powershell
.\executar.ps1 -InputVideo "URL_RTSP" -Show -MaxFps 20
.\executar.ps1 -InputVideo "URL_RTSP" -Show -MaxFps 10
```

O limitador espera antes de ler a imagem mais recente. A captura continua
consumindo o stream, evitando acumular atraso. Reduzir para 10 FPS poupa trabalho
de inferência e diminui a frequência de acompanhamento dos movimentos; a
decodificação do stream continua no ritmo da câmera. Arquivos mantêm todos os
quadros e são processados sem esse limite de tempo real.

`eventos.jsonl` inclui `processing_fps`; `resumo.json` registra o teto efetivo
em `processing_fps_limit`, além de `average_fps`.

## Decodificação, prévia e medição de desempenho

A captura FFmpeg ao vivo usa `-CaptureThreads 1`. No teste com esta câmera,
a escolha automática abriu 12 threads e entregou muitos quadros em grupos.
Como a captura guarda apenas o quadro mais recente para evitar atraso,
parte desses grupos era descartada antes da próxima inferência.
Uma thread reduziu esse efeito. `-CaptureThreads 0` restaura a escolha
automática; arquivos de vídeo e webcams DirectShow mantêm seu decodificador
habitual. O número efetivamente usado pelo FFmpeg aparece em
`capture.decoder_threads` no resumo.

A prévia usa `-PreviewMaxSide 1280` e também cabe na área disponível do monitor.
Quando a gravação está desligada, o desenho é feito diretamente nessa imagem
menor. As caixas de pessoas, faces, pontos de pose e linha são convertidos
somente para o desenho. YOLO, reconhecimento facial, tracking, contagem e
eventos continuam usando a imagem original. Com `-SaveVideo`, o desenho e
o MP4 conservam a resolução original e apenas a imagem da janela é reduzida.
`-PreviewMaxSide 0` permite enviar a imagem completa para a janela.
A leitura de teclas usa `pollKey` para processar os eventos sem a espera
adicional de `waitKey` no Windows. Q e Esc continuam encerrando a execução.

```powershell
.\executar.ps1 -InputVideo "URL_RTSP" -Show -MaxFps 20 -CaptureThreads 1 -PreviewMaxSide 1280
```

O teste final de 07/10/2026 usou o stream real de 2688 × 1520, a mesma engine
YOLO11L FP16 e 100 quadros por configuração, em uma cena sem pessoas.
A aplicação da câmera permaneceu aberta em outro processo. As chamadas
de criação, exibição e teclado da janela foram substituídas durante o teste;
os resultados medem captura, análise e renderização da prévia.

| Configuração | FPS médio |
| --- | ---: |
| Decodificação automática e desenho na imagem completa | 12,24 |
| Uma thread de decodificação e prévia menor | 15,25 |
| Repetição da configuração automática | 11,64 |

Em uma comparação anterior de 80 quadros por configuração, apenas mudar
o decodificador de automático para uma thread passou de 11,05 para 14,38 FPS,
com 14,29 FPS na repetição. A variação entre rodadas reforça a necessidade de
medir a execução completa depois de reiniciar a aplicação com os novos padrões.
As medições estão em `outputs/diagnostico_pipeline_completo_20261007/`.

`eventos.jsonl` e `resumo.json` agora distinguem:

- `capture.received_fps`: ritmo de chegada/decodificação na última janela de
  aproximadamente um segundo, separado do FPS informado nos metadados.
- `capture.frames_received`, `frames_delivered` e `frames_dropped`: quantos
  quadros chegaram, foram entregues ao processamento e foram substituídos
  por um quadro mais recente antes de serem consumidos.
- `timings_ms` / `average_timings_ms`: espera pelo limitador, espera na captura,
  rotação, YOLO, tracking, acesso/processamento facial, desenho, gravação e janela.
- `average_timings_ms.event_write` e `cycle`: escrita do evento e duração total
  de cada ciclo. O campo histórico `frame` mede apenas a análise e se sobrepõe
  a outras etapas; ele não deve ser somado a elas.
- `yolo_timings_ms` / `average_yolo_timings_ms`: pré-processamento, inferência
  e pós-processamento informados pelo Ultralytics.

O tempo de `capture_wait` inclui a espera por um quadro novo. Na modalidade
facial assíncrona, o trabalho do worker é medido em `face_processing`; a etapa
`access_and_faces` mede a coleta dos resultados, o agendamento e o controle
de acesso no loop principal. O FPS da janela inclui o tempo entre atualizações,
enquanto `cycle` ajuda a identificar qual etapa está limitando o ritmo.

## Pausas na captura e eventos da janela

Depois de abrir a janela, a espera por um quadro novo da câmera processa eventos
do OpenCV a cada aproximadamente 20 ms. Isso permite mover a janela e encerrar
com Q ou Esc mesmo quando o leitor estiver aguardando o stream ou reconectando.
O callback de eventos roda fora do bloqueio da captura para o leitor continuar
publicando quadros. Ao encerrar, a janela é destruída antes da espera pela
finalização do decodificador. Arquivos de vídeo mantêm a leitura sequencial.

Durante essa espera, o YOLO, o tracking e a contagem só processam um quadro
quando uma imagem nova for recebida. A imagem exibida conserva o último quadro;
a resposta da janela e a fluidez do stream são medições diferentes.

A captura agora registra `capture.delivered_frame` com o número de sequência,
tipo de quadro (`I`, `P`, `B` ou `unknown`), `native_read_ms` e
`arrival_interval_ms`. Esses dados pertencem ao quadro entregue ao processamento,
mesmo se o leitor receber outras imagens durante a inferência.
`capture.native_read_stalls_ge_200ms` conta leituras de pelo menos 200 ms por tipo
de quadro, incluindo imagens descartadas antes da inferência. O tempo nativo
inclui a chegada dos dados e a decodificação; ele não mede somente a CPU.

Na Intelbras VIP 3430 B IA conectada por cabo, a configuração informada em
07/10/2026 foi H.264H, 30 FPS, compressão inteligente desligada, CBR de 4096 kbit/s
e intervalo I de 60 quadros. Isso produz um I-frame a cada 2 segundos.
O teste isolado de captura encontrou leituras demoradas no último P-frame e no
I-frame seguinte. A comparação manteve a configuração da câmera e a aplicação
original em execução, sem inferência, exibição ou comandos ao ESP32:

| Transporte | Tempo medido | Leituras de P-frames ≥ 200 ms | Leituras de I-frames ≥ 200 ms |
| --- | ---: | ---: | ---: |
| TCP | 12 s | 6 | 6 |
| UDP | 16 s | 8 | 8 |

A periodicidade permaneceu nos dois transportes. Esses resultados localizam
as pausas na passagem entre GOPs, mas ainda exigem separar o comportamento
do encoder e do decodificador para escolher a correção da entrega do vídeo.
As medições estão em `outputs/diagnostico_travadas_20261007/probe_iframes.json`,
`probe_tcp.json` e `probe_udp.json`, com os eventos correspondentes.
O [manual da VIP 3430 B IA](https://backend.intelbras.com/sites/default/files/2025-09/Manual_VIP_3430_BD_IA_03-25_site.pdf)
descreve os ajustes de codec, taxa de bits e intervalo do frame I.

No teste seguinte, reduzir o intervalo I de 60 para 30 tornou as pausas mais
frequentes e mais curtas. A leitura de pacotes comprimidos, sem decodificação,
também apresentou esperas periódicas; portanto, havia atraso antes da inferência
e da decodificação. Esse teste não separa entrega pela câmera, rede e buffering
do RTSP. Os resultados estão em `probe_gop30.json`, `probe_gop30_raw.json` e
`comparacao_gop60_gop30.json`, na mesma pasta de diagnóstico.

O usuário relatou que a troca de CBR para VBR eliminou as pausas perceptíveis.
A configuração mostrada foi H.264H, 1920×1080, 30 FPS, VBR, qualidade 4,
4096 kbit/s, intervalo I de 30 e compressão inteligente desligada.
A captura isolada de 25 segundos confirmou resolução 1920×1080 e aproximadamente
30 FPS. Descartados os dois primeiros segundos de aquecimento, a mediana da
leitura foi 32,862 ms, o percentil 99 foi 89,487 ms e o máximo foi 98,124 ms,
sem leituras de pelo menos 100 ms. O relatório é `probe_vbr_gop30.json`.
Isso valida a fluidez da captura nessa amostra, sem medir o FPS do pipeline
completo. A resolução inicial de outro diagnóstico era 2688×1520; por isso,
comparações com aquele diagnóstico também incluem a diferença de resolução.
As configurações da câmera foram alteradas pelo usuário; os testes apenas
leram o stream e não reiniciaram a aplicação nem enviaram comandos ao ESP32.

## Linha de entrada e saída

O monitor separado usa YOLO + OC-SORT e verifica o cruzamento pelo centro
de cada caixa, indicado pelo ponto amarelo:

```powershell
.\monitorar_linha.ps1 -InputVideo "URL_RTSP"
```

Clique em dois pontos para desenhar a linha. Pressione `R` para inverter os
rótulos entrada/saída, `C` para redefinir a linha e `Q` ou `Esc` para encerrar.
A linha padrão usa as coordenadas normalizadas
`[0.489844, 0.155556, 0.500391, 0.890972]`.
A configuração normalizada é salva em `outputs/linha/linha.json`, e cada
cruzamento é registrado em `outputs/linha/cruzamentos.jsonl`. A faixa de
histerese padrão de 18 pixels evita contagens repetidas quando o tracking oscila
sobre a linha.

## Acessos e ESP32 no executar.ps1

O `executar.ps1` também desenha e monitora a linha, carregando
`outputs/linha/linha.json`. Se esse arquivo não existir, usa a linha padrão
`[0.489844, 0.155556, 0.500391, 0.890972]`. O centro da caixa deve
atravessar o segmento e ultrapassar a faixa de histerese para contar. A seta
`ENTRADA` mostra o sentido; nessa configuração, entrada é da direita para a
esquerda da imagem. Use `-InvertLine` se o sentido físico for o contrário.
As coordenadas se aplicam à imagem já rotacionada por `-Rotate`.

```powershell
.\executar.ps1 -InputVideo "URL_RTSP" -Show -EspUrl "http://192.168.3.1"
```

O banco persistente fica em `data/acessos.sqlite3`:

- `acessos`: nome, horário de entrada, horário de saída, permanência em segundos,
  IDs de tracking e execução. Cada identidade pode ter uma visita aberta.
- `cruzamentos`: todos os cruzamentos, incluindo desconhecidos e saídas sem
  uma entrada anterior, com direção, horário, identidade e resultado do registro.

Uma entrada reconhecida abre uma visita. Uma saída reconhecida fecha a visita
aberta pelo **nome**, mesmo que o tracking seja outro. Sumir da imagem mantém a
visita aberta. Reconhecimento sem cruzar a linha não registra entrada nem saída.
Se a face for reconhecida depois do cruzamento, enquanto o tracking ainda
estiver ativo, o evento é associado à identidade usando o horário original.
Visitas abertas persistem quando o programa reinicia. Uma saída sem visita
aberta é apenas auditada, pois não há horário de entrada para calcular duração.

Os horários incluem o fuso local do computador. Em RTSP/webcam é usado o
relógio do sistema; em arquivos, a linha do tempo do vídeo avança a partir do
início da execução, para a permanência não depender da velocidade da inferência.
Para testar com arquivos sem misturar visitas reais, use outro banco:

```powershell
.\executar.ps1 -InputVideo "C:\videos\teste.mp4" -Show -NoEsp -Database "data\teste.sqlite3"
```

O firmware `esp32c3_leds_wifi_epi_original/main.py` já atende os comandos
`GET /gpioN:on` e `GET /gpioN:off`, além de `POST /api/state` com
`{"gate_action": "on"}` ou `{"gate_action": "off"}` para o relé do portão
na GPIO1. Não é necessário alterá-lo para esses comandos:

| Situação | GPIO4 buzzer | GPIO5 laranja | GPIO6 vermelho | GPIO7 verde |
| --- | --- | --- | --- | --- |
| Nenhuma pessoa visível | desligado* | ligado | desligado | desligado |
| Alguma pessoa desconhecida | desligado* | desligado | ligado | desligado |
| Todas as pessoas visíveis reconhecidas | desligado* | desligado | desligado | ligado |
| Desconhecido cruza a linha | ligado por 3 s | conforme situação | conforme situação | conforme situação |

O portão fica bloqueado com sinal laranja ou vermelho. Somente o sinal verde
(ao menos uma pessoa visível e todas reconhecidas) libera o portão, mantendo
o relé ligado enquanto essa condição durar. A ligação física deve usar
relé desligado para bloquear e relé ligado para liberar.
O cliente confirma o LED verde antes de liberar e envia o bloqueio antes
de trocar as luzes quando a autorização termina. A resposta `gate_state`
é verificada; comandos ignorados pelo debounce de 350 ms do firmware são
repetidos até confirmar o estado solicitado.

\* Um alerta iniciado continua até terminar sua duração, mesmo que a pessoa
saia da imagem ou seja reconhecida depois. Novos cruzamentos desconhecidos
estendem o alerta. Use `-AlarmSeconds 5` para mudar sua duração.

O endereço padrão é `http://192.168.3.1`, definido no firmware em modo AP.
Use `-EspUrl` para outro endereço e `-NoEsp` para desativar a comunicação.
O computador precisa alcançar o ESP e a câmera na rede. Uma thread envia os
comandos, apaga as outras cores antes de ligar a escolhida, tenta reconectar
após falhas e sincroniza o estado a cada 5 segundos. A janela mostra o estado
da conexão. Ao encerrar normalmente ou com Ctrl+C, o programa solicita buzzer
desligado, laranja ligado e portão bloqueado, repetindo o bloqueio por um
tempo limitado se necessário. Isso depende de o ESP estar acessível.
O firmware atual mantém o último estado do relé se a comunicação cair ou
o processo for encerrado à força. Bloqueio automático nessa situação exige
um timeout de autorização no firmware; não é garantido pelo cliente Python.

`eventos.jsonl` inclui os cruzamentos e contadores. `resumo.json` inclui entradas,
saídas, cruzamentos desconhecidos, visitas abertas e caminho do banco. As
contagens de cruzamentos são por execução; as visitas permanecem no SQLite.
Para consultar as visitas em um cliente SQLite:

```sql
SELECT nome, horario_entrada, horario_saida, tempo_permanencia_segundos
FROM acessos ORDER BY horario_entrada DESC;
```
