(function () {
  "use strict";

  angular.module("inPhaseApp").controller("MainController", [
    "$location",
    "$rootScope",
    "$scope",
    "ApiService",
    function ($location, $rootScope, $scope, ApiService) {
      var vm = this;

      vm.apiBase = ApiService.getBaseUrl();
      vm.pageTitle = "Dashboard";
      vm.apiHealthy = false;

      vm.isActive = function (path) {
        return $location.path() === path;
      };

      vm.navigateTo = function (path) {
        $location.path(path);
        if (!$scope.$$phase) {
          $scope.$apply();
        }
      };

      var pageMap = {
        "/dashboard": "Project Dashboard",
        "/busi": "BUSI Explorer",
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
