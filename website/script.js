document.addEventListener('DOMContentLoaded', () => {
    const year = document.querySelector('#current-year');
    if (year) year.textContent = new Date().getFullYear();

    const tocLinks = [...document.querySelectorAll('.page-toc a')];
    const sections = tocLinks
        .map((link) => document.querySelector(link.getAttribute('href')))
        .filter(Boolean);

    if ('IntersectionObserver' in window && sections.length) {
        const linksById = new Map(
            tocLinks.map((link) => [link.getAttribute('href').slice(1), link])
        );

        const setActiveLink = (id) => {
            tocLinks.forEach((link) => {
                const active = link === linksById.get(id);
                link.classList.toggle('is-active', active);
                if (active) link.setAttribute('aria-current', 'location');
                else link.removeAttribute('aria-current');
            });
        };

        const observer = new IntersectionObserver((entries) => {
            const visible = entries
                .filter((entry) => entry.isIntersecting)
                .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
            if (visible[0]) setActiveLink(visible[0].target.id);
        }, { rootMargin: '-18% 0px -70% 0px', threshold: 0 });

        sections.forEach((section) => observer.observe(section));
    }

    const mobileContents = document.querySelector('.mobile-toc details');
    document.querySelectorAll('.mobile-toc a').forEach((link) => {
        link.addEventListener('click', () => {
            if (mobileContents) mobileContents.open = false;
        });
    });

    const loadFutureVideo = async (card) => {
        const source = card.dataset.videoSrc;
        const frame = card.querySelector('.future-media-frame');
        const status = card.querySelector('.coming-soon');
        if (!source || !frame) return;

        try {
            const response = await fetch(source, { method: 'HEAD', cache: 'no-store' });
            if (!response.ok) return;

            const video = document.createElement('video');
            video.controls = true;
            video.muted = true;
            video.playsInline = true;
            video.preload = 'metadata';
            video.src = source;

            video.addEventListener('loadedmetadata', () => {
                frame.replaceChildren(video);
                card.classList.add('media-ready');
                if (status) status.textContent = 'Recording available';
            }, { once: true });
        } catch {
            // Local file previews and absent future media keep the designed placeholder.
        }
    };

    document.querySelectorAll('.future-media').forEach(loadFutureVideo);
});
