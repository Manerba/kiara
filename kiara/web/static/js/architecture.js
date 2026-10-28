// Diagrams are pre-rendered SVGs. These controls need no network or renderer.
(function () {
    'use strict';

    function revealTarget() {
        const target = document.getElementById(window.location.hash.slice(1));
        if (!target) return;
        let parent = target;
        while (parent) {
            if (parent.tagName === 'DETAILS') parent.open = true;
            parent = parent.parentElement;
        }
        target.scrollIntoView({ block: 'start' });
    }

    document.querySelectorAll('[data-atlas-size]').forEach(function (button) {
        button.addEventListener('click', function () {
            const canvas = document.getElementById(button.getAttribute('aria-controls'));
            if (canvas) {
                const actual = button.dataset.atlasSize === 'actual';
                canvas.classList.toggle('atlas-actual-size', actual);
                canvas.classList.toggle('atlas-fit-size', !actual);
            }
        });
    });
    window.addEventListener('hashchange', revealTarget);
    revealTarget();
})();
