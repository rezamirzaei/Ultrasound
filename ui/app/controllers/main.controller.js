(function () {
  "use strict";

  angular.module("inPhaseApp").controller("MainController", [
    "$location",
    "$rootScope",
    "ApiService",
    function ($location, $rootScope, ApiService) {
      var vm = this;

      vm.apiBase = ApiService.getBaseUrl();
      vm.pageTitle = "Dashboard";
      vm.apiHealthy = false;

      vm.isActive = function (path) {
        return $location.path() === path;
      };

      var pageMap = {
        "/dashboard": "Project Dashboard",
        "/preprocessing": "Preprocessing Lab",
        "/ndt": "NDT Sample Explorer",
      };

      $rootScope.$on("$routeChangeSuccess", function () {
        vm.pageTitle = pageMap[$location.path()] || "Ultrasound Platform";
      });

      ApiService.health()
        .then(function () {
          vm.apiHealthy = true;
        })
        .catch(function () {
          vm.apiHealthy = false;
        });
    },
  ]);
})();
