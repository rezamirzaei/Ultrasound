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
      vm.waveformChart = null;
      vm.phaseChart = null;
      vm.residualChart = null;
      vm.form = {
        case_name: "Parietal_free_field_0_XY",
        window_length: 256,
        n_fft: 80,
        hop_length: 8,
        max_iterations: 120,
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
        vm.waveformChart = buildOverlayChart(preview.true_signal || [], preview.recovered_signal || []);
        vm.phaseChart = buildOverlayChart(
          preview.true_phase_spectrum || [],
          preview.recovered_phase_spectrum || []
        );
        vm.residualChart = buildLineChart(preview.residual_curve || [0]);
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
            vm.form.window_length = status.recommended_window_length || vm.form.window_length;
            vm.form.n_fft = status.recommended_n_fft || vm.form.n_fft;
            vm.form.hop_length = status.recommended_hop_length || vm.form.hop_length;
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
        vm.waveformChart = null;
        vm.phaseChart = null;
        vm.residualChart = null;

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
