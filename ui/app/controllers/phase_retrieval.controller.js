(function () {
  "use strict";

  angular.module("inPhaseApp").controller("PhaseRetrievalController", [
    "ApiService",
    function (ApiService) {
      var vm = this;

      vm.loading = true;
      vm.error = null;
      vm.previewError = null;
      vm.running = false;
      vm.status = null;
      vm.preview = null;
      vm.realChart = null;
      vm.imagChart = null;
      vm.rmseChart = null;
      vm.form = {
        case_name: "carotid_long",
        segment_length: 96,
        measurement_ratio: 5,
        max_iterations: 150,
        seed: 42,
      };

      vm.run = run;

      function buildPath(values, width, height, padX, padY, yMin, yMax) {
        var plotWidth = width - padX * 2;
        var plotHeight = height - padY * 2;
        var xStep = values.length > 1 ? plotWidth / (values.length - 1) : 0;
        var path = "";
        for (var i = 0; i < values.length; i += 1) {
          var x = padX + i * xStep;
          var y = padY + (1 - (values[i] - yMin) / (yMax - yMin || 1)) * plotHeight;
          path += (i === 0 ? "M " : " L ") + x.toFixed(2) + " " + y.toFixed(2);
        }
        return path;
      }

      function buildOverlayChart(primary, secondary) {
        var width = 920;
        var height = 240;
        var padX = 16;
        var padY = 16;
        var values = (primary || []).concat(secondary || []);
        var yMin = Math.min.apply(null, values);
        var yMax = Math.max.apply(null, values);
        if (!Number.isFinite(yMin) || !Number.isFinite(yMax) || Math.abs(yMax - yMin) < 1e-12) {
          yMin = -1;
          yMax = 1;
        }
        return {
          width: width,
          height: height,
          baselineY: padY + (1 - (0 - yMin) / (yMax - yMin)) * (height - padY * 2),
          primaryPath: buildPath(primary, width, height, padX, padY, yMin, yMax),
          secondaryPath: buildPath(secondary, width, height, padX, padY, yMin, yMax),
        };
      }

      function buildLineChart(values) {
        var width = 920;
        var height = 220;
        var padX = 16;
        var padY = 16;
        var yMin = Math.min.apply(null, values);
        var yMax = Math.max.apply(null, values);
        if (!Number.isFinite(yMin) || !Number.isFinite(yMax) || Math.abs(yMax - yMin) < 1e-12) {
          yMin = 0;
          yMax = 1;
        }
        return {
          width: width,
          height: height,
          path: buildPath(values, width, height, padX, padY, yMin, yMax),
        };
      }

      function renderCharts(preview) {
        vm.realChart = buildOverlayChart(preview.true_real || [], preview.recovered_real || []);
        vm.imagChart = buildOverlayChart(preview.true_imag || [], preview.recovered_imag || []);
        vm.rmseChart = buildLineChart(preview.amplitude_rmse_curve || [0]);
      }

      function refreshStatus() {
        return ApiService.getPhaseRetrievalStatus()
          .then(function (status) {
            vm.status = status;
            if (status.available_cases && status.available_cases.length) {
              if (status.available_cases.indexOf(vm.form.case_name) === -1) {
                vm.form.case_name = status.recommended_case || status.available_cases[0];
              }
            }
            vm.form.segment_length = status.recommended_segment_length || vm.form.segment_length;
            vm.form.measurement_ratio = status.recommended_measurement_ratio || vm.form.measurement_ratio;
          })
          .catch(function (error) {
            vm.error = error.detail || "Failed to load phase retrieval status";
            vm.status = null;
          });
      }

      function run() {
        vm.running = true;
        vm.previewError = null;
        vm.preview = null;
        vm.realChart = null;
        vm.imagChart = null;
        vm.rmseChart = null;

        ApiService.previewPhaseRetrieval(vm.form)
          .then(function (preview) {
            vm.preview = preview;
            renderCharts(preview);
          })
          .catch(function (error) {
            vm.previewError = error.detail || "Failed to run phase retrieval preview";
          })
          .finally(function () {
            vm.running = false;
          });
      }

      function init() {
        refreshStatus()
          .then(function () {
            if (vm.status && vm.status.dataset_available) {
              run();
            }
          })
          .finally(function () {
            vm.loading = false;
          });
      }

      init();
    },
  ]);
})();
