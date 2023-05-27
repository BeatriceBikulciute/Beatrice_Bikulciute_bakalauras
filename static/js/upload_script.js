const uploadForm = document.querySelector('#uploadForm');
const selectedFile = document.querySelector('#selectedFile');

// Įraso faila kai jis yra pasirinktas
function updateFileName() {
  document.querySelector('.custom-file-input').addEventListener('change', function (e) {
    var fileName = e.target.files[0].name;
    var nextSibling = e.target.nextElementSibling;
    nextSibling.innerText = fileName;
  })
}

updateFileName();

// Failo įkėlimo klaidų kontrolė
uploadForm.addEventListener('submit', function(event){
  const fileName = selectedFile.value.split('\\').pop();
  const fileExt = fileName.split('.').pop().toLowerCase();

  if (!fileName) {
    event.preventDefault();
    selectedFile.classList.add('is-invalid');
    selectedFile.nextElementSibling.textContent = 'Nepasirinkote failo.';
    return;
  }

  if (fileExt !== 'csv') {
    event.preventDefault();
    selectedFile.classList.add('is-invalid');
    selectedFile.nextElementSibling.textContent = 'Pasirinkite .csv failą.';
    return;
  }

  selectedFile.classList.remove('is-invalid');
  selectedFile.nextElementSibling.textContent = 'Pasirinkite failą';
});
