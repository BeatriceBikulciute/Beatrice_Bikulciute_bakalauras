const dropdown = document.getElementById("dropdown");
const sarimaxInput = document.getElementById("sarimax_num");
const arInput = document.getElementById("ar");
const maInput = document.getElementById("ma");
const sarInput = document.getElementById("sar");
const smaInput = document.getElementById("sma");
const epochInput = document.getElementById("epoch");
const neurInput = document.getElementById("neur_sk");
const trendInput = document.getElementById("trend_type");

// Sarimax formos įjungimas ir išjungimas
dropdown.addEventListener("change", function() {
  if (dropdown.value === "option_sarimax") {
    sarimaxInput.removeAttribute("disabled");
  } else {
    sarimaxInput.setAttribute("disabled", true);
  }
});

// AR formos įjungimas ir išjungimas SARIMAX
dropdown.addEventListener("change", function() {
  if (dropdown.value === "option_arima" || dropdown.value === "option_sarimax"){
    arInput.removeAttribute("disabled");
  } else {
    arInput.setAttribute("disabled", true);
  }
});

// MA formos įjungimas ir išjungimas SARIMAX
dropdown.addEventListener("change", function() {
  if (dropdown.value === "option_arima" || dropdown.value === "option_sarimax") {
    maInput.removeAttribute("disabled");
  } else {
    maInput.setAttribute("disabled", true);
  }
});

// SAR formos įjungimas ir išjungimas SARIMAX
dropdown.addEventListener("change", function() {
  if (dropdown.value === "option_sarimax"){
    sarInput.removeAttribute("disabled");
  } else {
    sarInput.setAttribute("disabled", true);
  }
});

// MA formos įjungimas ir išjungimas SARIMAX
dropdown.addEventListener("change", function() {
  if (dropdown.value === "option_sarimax") {
    smaInput.removeAttribute("disabled");
  } else {
    smaInput.setAttribute("disabled", true);
  }
});

// LSTM epoch
dropdown.addEventListener("change", function() {
  if (dropdown.value === "option_lstm"){
    epochInput.removeAttribute("disabled");
  } else {
    epochInput.setAttribute("disabled", true);
  }
});

// LSTM neur_sk
dropdown.addEventListener("change", function() {
  if (dropdown.value === "option_lstm"){
    neurInput.removeAttribute("disabled");
  } else {
    neurInput.setAttribute("disabled", true);
  }
});
// Trend type
dropdown.addEventListener("change", function() {
  if (dropdown.value === "option_dlt"){
    trendInput.removeAttribute("disabled");
  } else {
    trendInput.setAttribute("disabled", true);
  }
});