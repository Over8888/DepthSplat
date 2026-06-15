import { HistoryOutlined, PlusOutlined, ReadOutlined, SettingOutlined } from '@ant-design/icons';
import { Layout, Menu, Typography } from 'antd';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';

const { Header, Content, Sider } = Layout;

const items = [
  { key: '/tasks/new', icon: <PlusOutlined />, label: '\u65b0\u5efa\u4efb\u52a1' },
  { key: '/history', icon: <HistoryOutlined />, label: '\u5386\u53f2\u8bb0\u5f55' },
  { key: '/result', icon: <ReadOutlined />, label: '\u7ed3\u679c\u67e5\u770b' },
  { key: '/settings', icon: <SettingOutlined />, label: '\u8bbe\u7f6e' },
];

export function AppLayout() {
  const location = useLocation();
  const navigate = useNavigate();
  const selectedKey =
    location.pathname.startsWith('/tasks/') && location.pathname !== '/tasks/new'
      ? '/history'
      : location.pathname;

  return (
    <Layout className="app-shell">
      <Sider breakpoint="lg" collapsedWidth="0" className="app-sider">
        <div className="brand">三维重建系统</div>
        <Menu theme="dark" mode="inline" selectedKeys={[selectedKey]} items={items} onClick={({ key }) => navigate(key)} />
      </Sider>
      <Layout className="app-main">
        <Header className="app-header">
          <Typography.Title level={4} style={{ margin: 0 }}>
            {'任务控制台'}
          </Typography.Title>
        </Header>
        <Content className="app-content">
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  );
}
