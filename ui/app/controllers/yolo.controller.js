(function () {
  "use strict";

  angular.module("inPhaseApp").controller("YoloController", [
    "ApiService",
    "$q",
    function (ApiService, $q) {
      var vm = this;

      vm.loading = true;
      vm.error = null;
      vm.status = null;

      vm.sampleLoading = false;
      vm.sampleError = null;
      vm.sample = null;

      vm.predictionLoading = false;
      vm.predictionError = null;
      vm.prediction = null;

      vm.trainingLoading = false;
      vm.trainingError = null;
      vm.trainingResult = null;
      vm.canTrain = ApiService.hasRole("analyst");

      vm.sampleForm = {
        category: "Benign",
        sample_index: 0,
      };

      vm.predictForm = {
        model: "yolo11n.pt",
        confidence: 0.25,
        iou_threshold: 0.45,
        image_size: 640,
        max_detections: 100,
      };

      vm.trainForm = {
        epochs: 30,
        batch_size: 16,
        image_size: 640,
        learning_rate: 0.01,
        patience: 10,
        freeze_layers: 10,
        pretrained_weights: "yolo11n.pt",
      };

      vm.refreshStatus = refreshStatus;
      vm.loadSample = loadSample;
      vm.runPrediction = runPrediction;
      vm.startTraining = startTraining;

      function refreshStatus() {
        return ApiService.getLiverYoloStatus()
          .then(function (payload) {
            vm.status = payload;
            // Auto-select trained weights when available.
            if (payload.trained_weights) {
              vm.predictForm.model = payload.default_model;
            }
          })
          .catch(function (error) {
            vm.status = null;
            vm.error = error.detail || "Failed to load liver YOLO lab status";
          });
      }

      function loadSample() {
        vm.sampleLoading = true;
        vm.sampleError = null;
        vm.sample = null;
        vm.prediction = null;
        vm.predictionError = null;

        var category = vm.sampleForm.category || "Benign";
        var index = Math.max(0, Math.floor(vm.sampleForm.sample_index || 0));

        return ApiService.getLiverSample(category, index)
          .then(function (payload) {
            vm.sample = payload;
          })
          .catch(function (error) {
            vm.sampleError = error.detail || "Failed to load liver sample";
          })
          .finally(function () {
            vm.sampleLoading = false;
          });
      }

      function runPrediction() {
        if (!vm.sample) {
          return;
        }
        vm.predictionLoading = true;
        vm.predictionError = null;
        vm.prediction = null;

        var category = vm.sample.category;
        var index = vm.sample.sample_index;

        var payload = {
          model: vm.predictForm.model,
          confidence: Number(vm.predictForm.confidence || 0.25),
          iou_threshold: Number(vm.predictForm.iou_threshold || 0.45),
          image_size: Math.max(160, Math.floor(vm.predictForm.image_size || 640)),
          max_detections: Math.max(1, Math.floor(vm.predictForm.max_detections || 100)),
        };

        return ApiService.predictLiverSample(category, index, payload)
          .then(function (prediction) {
            vm.prediction = prediction;
          })
          .catch(function (error) {
            vm.predictionError = error.detail || "Prediction failed";
          })
          .finally(function () {
            vm.predictionLoading = false;
          });
      }

      function startTraining() {
        if (!vm.canTrain) {
          return;
        }
        vm.trainingLoading = true;
        vm.trainingError = null;
        vm.trainingResult = null;

        var payload = {
          epochs: Math.max(1, Math.floor(vm.trainForm.epochs || 30)),
          batch_size: Math.max(1, Math.floor(vm.trainForm.batch_size || 16)),
          image_size: Math.max(160, Math.floor(vm.trainForm.image_size || 640)),
          learning_rate: Number(vm.trainForm.learning_rate || 0.01),
          patience: Math.max(1, Math.floor(vm.trainForm.patience || 10)),
          freeze_layers: Math.max(0, Math.floor(vm.trainForm.freeze_layers || 10)),
          pretrained_weights: vm.trainForm.pretrained_weights || "yolo11n.pt",
        };

        return ApiService.trainLiverYolo(payload)
          .then(function (result) {
            vm.trainingResult = result;
            // Refresh status to pick up new trained weights.
            return refreshStatus();
          })
          .catch(function (error) {
            vm.trainingError = error.detail || "Training failed";
          })
          .finally(function () {
            vm.trainingLoading = false;
          });
      }

      function init() {
        vm.loading = true;
        vm.error = null;
        $q.all([refreshStatus()])
          .then(function () {
            return loadSample();
          })
          .finally(function () {
            vm.loading = false;
          });
      }

      init();
    },
  ]);
})();
