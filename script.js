// Variables globales
let canvas, ctx, session;
const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

document.addEventListener("DOMContentLoaded", async () => {
  // --- 1. Gestion du Dessin ---
  canvas = document.getElementById("drawingCanvas");
  ctx = canvas.getContext("2d");
  let isDrawing = false;

  // Configuration pour ressembler à MNIST (Blanc sur Noir)
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "white";
  ctx.lineWidth = 20;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  // Souris
  canvas.addEventListener("mousedown", () => (isDrawing = true));
  canvas.addEventListener("mouseup", () => {
    isDrawing = false;
    ctx.beginPath();
  });
  canvas.addEventListener("mousemove", draw);

  // Touch (Mobile/Tablette)
  canvas.addEventListener("touchstart", (e) => {
    isDrawing = true;
    e.preventDefault();
  });
  canvas.addEventListener("touchend", () => {
    isDrawing = false;
    ctx.beginPath();
  });
  canvas.addEventListener("touchmove", drawTouch);

  function draw(e) {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
  }

  function drawTouch(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const touch = e.touches[0];
    ctx.lineTo(touch.clientX - rect.left, touch.clientY - rect.top);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(touch.clientX - rect.left, touch.clientY - rect.top);
  }

  // --- 2. Chargement du modèle ONNX ---
  async function loadModel() {
    document.getElementById("loading").style.display = "block";
    try {
      // 1. On télécharge le fichier manuellement
      const response = await fetch("./mon_modele_v2.onnx");

      if (!response.ok) {
        throw new Error("Impossible de trouver le fichier mon_modele_v2.onnx");
      }

      // 2. On convertit la réponse en 'buffer' (données brutes en mémoire)
      const arrayBuffer = await response.arrayBuffer();

      // 3. CORRECTION ICI : On donne le buffer à ONNX, pas le chemin du fichier !
      session = await ort.InferenceSession.create(arrayBuffer);

      document.getElementById("loading").style.display = "none";
      console.log("Modèle chargé avec succès via ArrayBuffer !");
    } catch (e) {
      alert("Erreur chargement modèle : " + e);
      console.error(e);
    }
  }
  loadModel();
});

// Fonctions globales accessibles depuis les boutons HTML
function clearCanvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById("pred-letter").innerText = "?";
  document.getElementById("pred-conf").innerText = "Confiance: 0%";
  ctx.beginPath();
}

async function predict() {
  if (!session) return;

  // 1. Réduire l'image à 28x28 pixels
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = 28;
  tempCanvas.height = 28;
  const tempCtx = tempCanvas.getContext("2d");
  tempCtx.drawImage(canvas, 0, 0, 28, 28);

  // 2. Extraire les données pixels
  const imageData = tempCtx.getImageData(0, 0, 28, 28);
  const data = imageData.data; // [R, G, B, A, R, G, B, A, ...]

  // 3. Convertir en Float32Array normalisé (comme PyTorch)
  // Entrée attendue : [1, 1, 28, 28] -> 784 valeurs float
  const inputData = new Float32Array(784);

  for (let i = 0; i < 784; i++) {
    // On prend juste le canal Rouge (puisque c'est noir et blanc, R=G=B)
    let pixelVal = data[i * 4];

    // Normalisation PyTorch : ToTensor() puis Normalize((0.5,), (0.5,))
    // ToTensor() : divise par 255 => range [0, 1]
    // Normalize((0.5,), (0.5,)) : (x - 0.5) / 0.5 => range [-1, 1]
    // Ici : fond noir => -1, trait blanc => +1 (même convention qu'EMNIST)
    inputData[i] = (pixelVal / 255.0 - 0.5) / 0.5;
  }

  console.log(inputData);

  // 4. Créer le tenseur ONNX
  const tensor = new ort.Tensor("float32", inputData, [1, 1, 28, 28]);

  // 5. Lancer l'inférence
  const results = await session.run({ input: tensor });
  const output = results.output.data;

  // Debug: inspect raw logits before post-processing
  console.log("Logits (output):", output);

  // 6. Calcul Softmax pour convertir logits → probabilités [0, 1]
  const exps = Array.from(output).map((x) => Math.exp(x));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  const probabilities = exps.map((exp) => exp / sumExps);

  // 7. Trouver la classe avec la probabilité maximale
  let maxProb = 0;
  let maxIndex = 0;
  for (let i = 0; i < probabilities.length; i++) {
    if (probabilities[i] > maxProb) {
      maxProb = probabilities[i];
      maxIndex = i;
    }
  }

  console.log("Probabilités (softmax):", probabilities);
  console.log(
    `Prédiction: ${alphabet[maxIndex]} avec confiance ${(maxProb * 100).toFixed(
      1
    )}%`
  );

  // Affichage
  document.getElementById("pred-letter").innerText = alphabet[maxIndex];
  document.getElementById("pred-conf").innerText = `Confiance: ${(
    maxProb * 100
  ).toFixed(1)}%`;
}
