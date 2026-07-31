// Vanilla JS only -- there is no build step in this project.
(function () {
  "use strict";

  // Only one rollout plays at a time, so the grid stays readable.
  document.addEventListener(
    "play",
    function (event) {
      document.querySelectorAll("video").forEach(function (other) {
        if (other !== event.target) {
          other.pause();
        }
      });
    },
    true
  );

  // Comparing a run with itself is never interesting; nudge B off A.
  var form = document.querySelector(".compare-picker");
  if (!form) {
    return;
  }
  var a = form.querySelector('select[name="a"]');
  var b = form.querySelector('select[name="b"]');
  function keepDistinct(changed, other) {
    if (changed.value && changed.value === other.value) {
      var alternative = Array.prototype.find.call(other.options, function (option) {
        return option.value && option.value !== changed.value;
      });
      other.value = alternative ? alternative.value : "";
    }
  }
  a.addEventListener("change", function () {
    keepDistinct(a, b);
  });
  b.addEventListener("change", function () {
    keepDistinct(b, a);
  });
})();
