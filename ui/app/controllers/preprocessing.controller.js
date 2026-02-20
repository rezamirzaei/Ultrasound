(function () {
  "use strict";

  angular.module("inPhaseApp").controller("PreprocessingController", [
    "ApiService",
    function (ApiService) {
      var vm = this;

      vm.form = {
        class_name: "benign",
        sample_index: 0,
        lambda_tv: 0.06,
        rho: 1.0,
        n_iter: 35,
        clip_limit: 2.5,
      };

      vm.loading = false;
      vm.error = null;
      vm.preview = null;

      vm.run = function () {
        vm.loading = true;
        vm.error = null;
        vm.preview = null;

        ApiService.previewPreprocessing(vm.form)
          .then(function (response) {
            vm.preview = response.data;
          })
          .catch(function (error) {
            vm.error =
              (error.data && error.data.detail) || "Failed to run preprocessing preview";
          })
          .finally(function () {
            vm.loading = false;
          });
      };

      vm.run();
    },
  ]);
})();
