var clickedImages = []; // Array untuk menyimpan gambar yang telah diklik
var score = 0; // Skor awal

document.querySelector(".row").addEventListener("click", function (event) {
  var clickedElement = event.target.closest(".game-img");
  if (clickedElement) {
    flipImage(clickedElement);
  }
});

function flipImage(element) {
  var imageId = element.querySelector("img").id;
  var newImage = element.getAttribute("data-new-image");
  var originalImage = element.getAttribute("data-ori");

  clickedImages.push(imageId); // Tambahkan ID gambar ke dalam array
  // Tukar gambar
  if (newImage) {
    element.querySelector("img").src = newImage;

    // Tambahkan class 'flipped' untuk memicu animasi
    element.querySelector("img").classList.add("flipped");

    // Setelah 1 detik, hapus class 'flipped'
    setTimeout(function () {
      element.querySelector("img").classList.remove("flipped");

      // Cek apakah ada dua gambar yang memiliki ID yang sama
      if (clickedImages.length === 2) {
        if (clickedImages[0] === clickedImages[1]) {
          // Jika ID kedua gambar sama, tambahkan skor
          score += 1;
          document.getElementById("score").textContent = score;
          console.log(clickedImages[0] + " " + clickedImages[1] + " Berhasil");

          // Reset array clickedImages setelah 2 gambar telah diklik
          clickedImages = [];
        } else {
          // Jika ID kedua gambar berbeda, kembalikan gambar ke aslinya
          setTimeout(function () {
            var prevElement1 = document.querySelector(
              "[id='" + clickedImages[0] + "']"
            ).parentElement;
            var prevElement2 = document.querySelector(
              "[id='" + clickedImages[1] + "']"
            ).parentElement;
            prevElement1.querySelector("img").src =
              prevElement1.getAttribute("data-ori");
            prevElement2.querySelector("img").src =
              prevElement2.getAttribute("data-ori");
            console.log(clickedImages[0] + " " + clickedImages[1] + " Gagal");
            clickedImages = [];
          }, 500);
        }
      }
    }, 500);
  } else {
    // Simpan gambar asli dalam atribut data-new-image
    element.setAttribute("data-new-image", element.querySelector("img").src);
  }
}

// Fungsi untuk menampilkan nama gambar pada elemen <p>
function showHint() {
  var gameImages = document.querySelectorAll(".col-md-3");

  gameImages.forEach(function (element) {
    var photoName = element.querySelector("p");
    photoName.style.display = "block"; // Tampilkan elemen <p>
  });
}

// Tambahkan event listener ke tombol "Hint"
document.getElementById("hintButton").addEventListener("click", showHint);

function surrender() {
  var gameImages = document.querySelectorAll(".col-md-3");

  gameImages.forEach(function (element) {
    var imgElement = element.querySelector("img");
    imgElement.src = element.getAttribute("data-new-image");
    var photoName = element.querySelector("p");
    photoName.style.display = "block"; // Tampilkan elemen <p>
  });

  // Reset array clickedImages
  clickedImages = [];

  alert("Yakin sudah menyerah!")
}

document.getElementById("surrenderButton").addEventListener("click", surrender);

function restart() {
  confirm("Yakin mau di restart?");
  location.reload();
}

document.getElementById("restartButton").addEventListener("click", restart);