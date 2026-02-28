import JarvisConnectionService from './JarvisConnectionService';

describe('JarvisConnectionService control-plane recovery discovery', () => {
  let initSpy;
  const originalPortRange = process.env.REACT_APP_LOADING_SERVER_PORT_RANGE;

  beforeEach(() => {
    initSpy = jest
      .spyOn(JarvisConnectionService.prototype, '_initializeAsync')
      .mockImplementation(() => {});
    localStorage.clear();
    delete window.Ironcliw_LOADING_SERVER_PORT;
    delete process.env.REACT_APP_LOADING_SERVER_PORT;
    delete process.env.REACT_APP_LOADING_SERVER_PORT_RANGE;
    window.history.pushState({}, '', '/');
  });

  afterEach(() => {
    initSpy.mockRestore();
    if (originalPortRange === undefined) {
      delete process.env.REACT_APP_LOADING_SERVER_PORT_RANGE;
    } else {
      process.env.REACT_APP_LOADING_SERVER_PORT_RANGE = originalPortRange;
    }
  });

  test('includes unified supervisor loading ports and loopback variants', () => {
    window.Ironcliw_LOADING_SERVER_PORT = 8080;

    const service = new JarvisConnectionService();
    service.backendUrl = 'http://localhost:8010';

    const candidates = service._getControlPlaneCandidates();

    expect(candidates).toContain('http://localhost:8080');
    expect(candidates).toContain('http://127.0.0.1:8080');
    expect(candidates).toContain('http://localhost:3001');
  });

  test('supports explicit loading-server port ranges from env', () => {
    process.env.REACT_APP_LOADING_SERVER_PORT_RANGE = '9100-9101,9200';

    const service = new JarvisConnectionService();
    service.backendUrl = 'http://localhost:8010';

    const candidates = service._getControlPlaneCandidates();

    expect(candidates).toContain('http://localhost:9100');
    expect(candidates).toContain('http://localhost:9101');
    expect(candidates).toContain('http://localhost:9200');
  });
});
